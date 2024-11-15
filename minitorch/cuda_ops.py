# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")

# change to a cuda function
def device_jit(fn: Fn, **kwargs) -> Fn:
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn, **kwargs) -> FakeCUDAKernel:
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            # decide how many block in a grid
            # + THREADS_PER_BLOCK - 1 cuz we need to round up to the nearest integer
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            # 1D block with 1D thread
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            # it's for cuda reduce so out shape will be block size. It's more efficent for cuda to do reduce
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # to store the tmp index
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        # to store the tmp index
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        # cuda.blockDim.x = threadperblock
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        # Guard
        if i < out_size:
            # change i to the outindex, for example 0 -> (0,0,0) 
            to_index(i, out_shape, out_index)
            # reverse the broadcast index to orginal index
            broadcast_index(out_index, out_shape, in_shape, in_index)
            in_pos = index_to_position(in_index, in_strides)
            # map the input position to out put
            out[i] = fn(in_storage[in_pos])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        # Guard
        if i < out_size:
            # change i to the outindex, for example 0 -> (0,0,0) 
            to_index(i, out_shape, out_index)
            # reverse the broadcast index to corresponding index a
            broadcast_index(out_index, out_shape, a_shape, a_index)
            # reverse the broadcast index to corresponding index in b
            broadcast_index(out_index, out_shape, b_shape, b_index)
            out[i] = fn(
                # get the position in a and b from their index
                a_storage[index_to_position(a_index, a_strides)],
                b_storage[index_to_position(b_index, b_strides)]
            )
    # compile low level function to cuda kernel function
    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""Practice sum kernel to prepare for reduce.

    Given an array of length n and out of size n // blockDIM
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # TODO: Implement for Task 3.3.
    # evert block has its own share memeory. 
    # for example, a = 0~64. 0~31 will be put into block0's cache and 32~64 will be put into block1' cache 
    cache[pos] = a[i] if i < size else 0.0
    # wait till every thread in "a block" finish assignment
    cuda.syncthreads()

    # Perform reduction in shared memory
    stride = 1
    while stride < BLOCK_DIM:
        # for sum the cache in every block 
        # for example 
        # first loop : [1, 2, 3, 4, 5, 6, 7, 8]
        # second loop : [1+2, 2, 3+4, 4, 5+6, 6, 7+8, 8]
        # last loop : [10+26, 2, 7, 4, 26, 6, 15, 8]
        # cache[0] will be the sum of the first block
        if pos % (2 * stride) == 0:
            cache[pos] += cache[pos + stride]
        stride *= 2
        # wait every thread in a block finish their first loop
        cuda.syncthreads()

    # Write result from each block to global memory
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]

jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        # blocks represnt the position in output tensor
        # threads deal with each value in "dim" in a (input tenosr)

        # how many thread in a block
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        # im still debugging
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        if out_pos < out_size:
            # compute out_index from out_pos
            to_index(out_pos, out_shape, out_index)

            # get the block index along reduce_dim
            reduce_block_idx = out_index[reduce_dim]

            # compute reduce_start and reduce_end
            reduce_start = reduce_block_idx * BLOCK_DIM
            reduce_end = min(reduce_start + BLOCK_DIM, a_shape[reduce_dim])
            # compute the number of active threads for this block
            active_threads = reduce_end - reduce_start

            # a_index will be same as out index, except for reduce dim
            for d in range(len(a_shape)):
                a_index[d] = out_index[d]

            # defualt value will be different depend on mul reduce or add reduce
            acc = reduce_value

            # active thread
            if pos < active_threads:
                # idx = the position in reduce dim this thread dealing w
                idx = reduce_start + pos
                a_index[reduce_dim] = idx

                # get the value of this index
                a_pos = index_to_position(a_index, a_strides)
                acc = a_storage[a_pos]

            else:
                # For threads beyond active_threads, set acc to default value
                acc = reduce_value

            # sharing cache store the result of each thread in current block
            # ex reduce dim 1 (0,0) (0,1) ... (0, 1024) will store in cache [0] [1] [2]
            cache[pos] = acc
            # wait every thread 
            cuda.syncthreads()

            # Perform reduction in shared memory, using binary reduction
            stride = 1
            while stride < BLOCK_DIM:
            # for sum the cache in every block 
            # for example 
            # first loop : [1, 2, 3, 4, 5, 6, 7, 8]
            # second loop : [1+2, 2, 3+4, 4, 5+6, 6, 7+8, 8]
            # last loop : [10+26, 2, 7, 4, 26, 6, 15, 8]
            # cache[0] will be the sum of the first block = the value after reduction
                index = 2 * stride * pos
                if index + stride < BLOCK_DIM and (index + stride) < active_threads:
                    cache[index] = fn(cache[index], cache[index + stride])
                cuda.syncthreads()
                stride *= 2

            # store the value in out position when threadid = 0 in each block 
            if pos == 0:
                out_pos = index_to_position(out_index, out_strides)
                out[out_pos] = cache[0]
        '''
        # Guard 
        if out_pos < out_size:
            # get the output index from output position
            to_index(out_pos, out_shape, out_index)
            # start value will be different depend on mul reduce or add reduce
            acc = reduce_value
            # a_shape[reduce_dim] = original size of this dim
            for j in range(a_shape[reduce_dim]):
                a_index = cuda.local.array(MAX_DIMS, numba.int32)
                
                # get a_index 
                # a_index will be same as out index, except for reduce dim
                for k in range(len(out_index)):
                    a_index[k] = out_index[k]
                a_index[reduce_dim] = j

                acc = fn(acc, a_storage[index_to_position(a_index, a_strides)])
            # Load each thread's result into shared memory
            cache[pos] = acc 
            cuda.syncthreads()

            # Parallel reduction within the block
            stride = 1
            while stride < BLOCK_DIM:
                # for sum the cache in every block 
                # for example 
                # first loop : [1, 2, 3, 4, 5, 6, 7, 8]
                # second loop : [1+2, 2, 3+4, 4, 5+6, 6, 7+8, 8]
                # last loop : [10+26, 2, 7, 4, 26, 6, 15, 8]
                # cache[0] will be the sum of the first block
                if pos % (2 * stride) == 0 and (pos + stride) < BLOCK_DIM:
                    cache[pos] = fn(cache[pos], cache[pos + stride])
                stride *= 2
                
                cuda.syncthreads()

            # Write the result of the reduction to the output
            if pos == 0:
                out[out_pos] = cache[0]
            '''
    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    r"""Practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.3.
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    
    row, col = cuda.threadIdx.y, cuda.threadIdx.x
    sum = 0.0

    for k in range(size // BLOCK_DIM):
        a_shared[row, col] = a[row * size + (col + k * BLOCK_DIM)]
        b_shared[row, col] = b[(row + k * BLOCK_DIM) * size + col]
        cuda.syncthreads()

        for j in range(BLOCK_DIM):
            sum += a_shared[row, j] * b_shared[j, col]
        cuda.syncthreads()

    out[row * size + col] = sum

jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    # TODO: Implement for Task 3.4.
    sum = 0.0

    for k in range((a_shape[-1] + BLOCK_DIM - 1) // BLOCK_DIM):
        if i < a_shape[-2] and k * BLOCK_DIM + cuda.threadIdx.y < a_shape[-1]:
            a_shared[cuda.threadIdx.x, cuda.threadIdx.y] = a_storage[i * a_strides[-2] + (k * BLOCK_DIM + cuda.threadIdx.y) * a_strides[-1]]
        else:
            a_shared[cuda.threadIdx.x, cuda.threadIdx.y] = 0.0

        if k * BLOCK_DIM + cuda.threadIdx.x < b_shape[-2] and j < b_shape[-1]:
            b_shared[cuda.threadIdx.x, cuda.threadIdx.y] = b_storage[(k * BLOCK_DIM + cuda.threadIdx.x) * b_strides[-2] + j * b_strides[-1]]
        else:
            b_shared[cuda.threadIdx.x, cuda.threadIdx.y] = 0.0

        cuda.syncthreads()

        for n in range(BLOCK_DIM):
            sum += a_shared[cuda.threadIdx.x, n] * b_shared[n, cuda.threadIdx.y]
        cuda.syncthreads()

    if i < out_shape[-2] and j < out_shape[-1]:
        out[i * out_strides[-2] + j * out_strides[-1]] = sum


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
