# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


## Task 3.1 & Task3.2
### The Diagnostics Output 

```text

MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/kadelin/Desktop/Cornell/MLE/mod3-cin-cout/minitorch/fast_ops.py (163)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/kadelin/Desktop/Cornell/MLE/mod3-cin-cout/minitorch/fast_ops.py (163) 
-------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                              | 
        out: Storage,                                                                      | 
        out_shape: Shape,                                                                  | 
        out_strides: Strides,                                                              | 
        in_storage: Storage,                                                               | 
        in_shape: Shape,                                                                   | 
        in_strides: Strides,                                                               | 
    ) -> None:                                                                             | 
        # TODO: Implement for Task 3.1.                                                    | 
        if list(out_shape) == list(in_shape) and list(out_strides) == list(in_strides):    | 
            for i in prange(len(out)):-----------------------------------------------------| #0
                out[i] = fn(in_storage[i])                                                 | 
            return                                                                         | 
        else:                                                                              | 
            for i in prange(len(out)):-----------------------------------------------------| #1
                out_index = np.empty(MAX_DIMS, dtype=np.int32)                             | 
                in_index = np.empty(MAX_DIMS, dtype=np.int32)                              | 
                to_index(i, out_shape, out_index)                                          | 
                broadcast_index(out_index, out_shape, in_shape, in_index)                  | 
                out_pos = index_to_position(out_index, out_strides)                        | 
                in_pos = index_to_position(in_index, in_strides)                           | 
                out[out_pos] = fn(in_storage[in_pos])                                      | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #0, #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/kadelin/Desktop/Cornell/MLE/mod3-cin-cout/minitorch/fast_ops.py (178) is 
hoisted out of the parallel loop labelled #1 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/kadelin/Desktop/Cornell/MLE/mod3-cin-cout/minitorch/fast_ops.py (179) is 
hoisted out of the parallel loop labelled #1 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: in_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/kadelin/Desktop/Cornell/MLE/mod3-cin-cout/minitorch/fast_ops.py (212)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/kadelin/Desktop/Cornell/MLE/mod3-cin-cout/minitorch/fast_ops.py (212) 
---------------------------------------------------------------------------|loop #ID
    def _zip(                                                              | 
        out: Storage,                                                      | 
        out_shape: Shape,                                                  | 
        out_strides: Strides,                                              | 
        a_storage: Storage,                                                | 
        a_shape: Shape,                                                    | 
        a_strides: Strides,                                                | 
        b_storage: Storage,                                                | 
        b_shape: Shape,                                                    | 
        b_strides: Strides,                                                | 
    ) -> None:                                                             | 
        # TODO: Implement for Task 3.1.                                    | 
        if list(out_shape) == list(a_shape) == list(b_shape) and list(     | 
            out_strides                                                    | 
        ) == list(a_strides) == list(b_strides):                           | 
            for i in prange(len(out)):-------------------------------------| #2
                out[i] = fn(a_storage[i], b_storage[i])                    | 
            return                                                         | 
        else:                                                              | 
            for i in prange(len(out)):-------------------------------------| #3
                out_index = np.empty(MAX_DIMS, dtype=np.int32)             | 
                a_index = np.empty(MAX_DIMS, dtype=np.int32)               | 
                b_index = np.empty(MAX_DIMS, dtype=np.int32)               | 
                to_index(i, out_shape, out_index)                          | 
                broadcast_index(out_index, out_shape, a_shape, a_index)    | 
                broadcast_index(out_index, out_shape, b_shape, b_index)    | 
                a_pos = index_to_position(a_index, a_strides)              | 
                b_pos = index_to_position(b_index, b_strides)              | 
                out[i] = fn(a_storage[a_pos], b_storage[b_pos])            | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/kadelin/Desktop/Cornell/MLE/mod3-cin-cout/minitorch/fast_ops.py (232) is 
hoisted out of the parallel loop labelled #3 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/kadelin/Desktop/Cornell/MLE/mod3-cin-cout/minitorch/fast_ops.py (233) is 
hoisted out of the parallel loop labelled #3 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: a_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/kadelin/Desktop/Cornell/MLE/mod3-cin-cout/minitorch/fast_ops.py (234) is 
hoisted out of the parallel loop labelled #3 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: b_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/kadelin/Desktop/Cornell/MLE/mod3-cin-cout/minitorch/fast_ops.py (266)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/kadelin/Desktop/Cornell/MLE/mod3-cin-cout/minitorch/fast_ops.py (266) 
-------------------------------------------------------------------|loop #ID
    def _reduce(                                                   | 
        out: Storage,                                              | 
        out_shape: Shape,                                          | 
        out_strides: Strides,                                      | 
        a_storage: Storage,                                        | 
        a_shape: Shape,                                            | 
        a_strides: Strides,                                        | 
        reduce_dim: int,                                           | 
    ) -> None:                                                     | 
        # TODO: Implement for Task 3.1.                            | 
        reduce_size = a_shape[reduce_dim]                          | 
        reduce_stride = a_strides[reduce_dim]                      | 
                                                                   | 
        for i in prange(len(out)):---------------------------------| #4
            out_index = np.empty(MAX_DIMS, dtype=np.int32)         | 
            to_index(i, out_shape, out_index)                      | 
            out_pos = index_to_position(out_index, out_strides)    | 
            a_pos = index_to_position(out_index, a_strides)        | 
            reduce_val = out[out_pos]                              | 
            for j in range(reduce_size):                           | 
                reduce_val = fn(reduce_val, a_storage[a_pos])      | 
                a_pos += reduce_stride                             | 
            out[out_pos] = reduce_val                              | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #4).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/kadelin/Desktop/Cornell/MLE/mod3-cin-cout/minitorch/fast_ops.py (280) is 
hoisted out of the parallel loop labelled #4 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/kadelin/Desktop/Cornell/MLE/mod3-cin-cout/minitorch/fast_ops.py (293)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/kadelin/Desktop/Cornell/MLE/mod3-cin-cout/minitorch/fast_ops.py (293) 
--------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                | 
    out: Storage,                                                                           | 
    out_shape: Shape,                                                                       | 
    out_strides: Strides,                                                                   | 
    a_storage: Storage,                                                                     | 
    a_shape: Shape,                                                                         | 
    a_strides: Strides,                                                                     | 
    b_storage: Storage,                                                                     | 
    b_shape: Shape,                                                                         | 
    b_strides: Strides,                                                                     | 
) -> None:                                                                                  | 
    """NUMBA tensor matrix multiply function.                                               | 
                                                                                            | 
    Should work for any tensor shapes that broadcast as long as                             | 
                                                                                            | 
    ```                                                                                     | 
    assert a_shape[-1] == b_shape[-2]                                                       | 
    ```                                                                                     | 
                                                                                            | 
    Optimizations:                                                                          | 
                                                                                            | 
    * Outer loop in parallel                                                                | 
    * No index buffers or function calls                                                    | 
    * Inner loop should have no global writes, 1 multiply.                                  | 
                                                                                            | 
                                                                                            | 
    Args:                                                                                   | 
    ----                                                                                    | 
        out (Storage): storage for `out` tensor                                             | 
        out_shape (Shape): shape for `out` tensor                                           | 
        out_strides (Strides): strides for `out` tensor                                     | 
        a_storage (Storage): storage for `a` tensor                                         | 
        a_shape (Shape): shape for `a` tensor                                               | 
        a_strides (Strides): strides for `a` tensor                                         | 
        b_storage (Storage): storage for `b` tensor                                         | 
        b_shape (Shape): shape for `b` tensor                                               | 
        b_strides (Strides): strides for `b` tensor                                         | 
                                                                                            | 
    Returns:                                                                                | 
    -------                                                                                 | 
        None : Fills in `out`                                                               | 
                                                                                            | 
    """                                                                                     | 
    assert a_shape[-1] == b_shape[-2]                                                       | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                  | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                  | 
                                                                                            | 
    # TODO: Implement for Task 3.2.                                                         | 
    out_batch_stride = out_strides[0] if len(out_shape) > 2 else 0                          | 
    row_stride_a = a_strides[1]                                                             | 
    col_stride_b = b_strides[2]                                                             | 
    # Outer loop over batch and output dimensions                                           | 
    for n in prange(out_shape[0]):  # Batch-------------------------------------------------| #5
        for i in range(out_shape[1]):  # Output rows                                        | 
            for j in range(out_shape[2]):  # Output columns                                 | 
                # Calculate the position in the output tensor                               | 
                out_pos = n * out_batch_stride + i * out_strides[1] + j * out_strides[2]    | 
                out[out_pos] = 0  # Initialize to zero for accumulation                     | 
                acc = 0.0                                                                   | 
                a_pos = n * a_batch_stride + i * row_stride_a                               | 
                b_pos = n * b_batch_stride + j * col_stride_b                               | 
                # Accumulate the dot product                                                | 
                for _ in range(a_shape[-1]):                                                | 
                    acc += a_storage[a_pos] * b_storage[b_pos]                              | 
                    a_pos += a_strides[2]                                                   | 
                    b_pos += b_strides[1]                                                   | 
                out[out_pos] = acc                                                          | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #5).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```