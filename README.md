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

## Task3.4
```bash
!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/timing.py
```
```text
{'fast': np.float64(0.014357169469197592), 'gpu': np.float64(0.012360572814941406)}
Running size 256
{'fast': np.float64(0.09304157892862956), 'gpu': np.float64(0.04480719566345215)}
Running size 512
{'fast': np.float64(0.9815650780995687), 'gpu': np.float64(0.19446508089701334)}
Running size 1024
{'fast': np.float64(7.970567146937053), 'gpu': np.float64(0.8498539129892985)}

Timing summary
Size: 64
    fast: 0.00327
    gpu: 0.00610
Size: 128
    fast: 0.01436
    gpu: 0.01236
Size: 256
    fast: 0.09304
    gpu: 0.04481
Size: 512
    fast: 0.98157
    gpu: 0.19447
Size: 1024
    fast: 7.97057
    gpu: 0.84985
```

plot by google sheet

![Timing Summary for Matrix Multiplication](./Timing%20Summary%20for%20Matrix%20Multiplication.png)

## Task3.5

### Small Model
#### Split
##### CPU
```bash
!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
```
```text
Epoch 0: loss 7.5551, correct 31, time 4.0024 seconds
Epoch 10: loss 5.3866, correct 35, time 1.5340 seconds
Epoch 20: loss 5.6178, correct 42, time 1.5334 seconds
Epoch 30: loss 4.6200, correct 43, time 1.9837 seconds
Epoch 40: loss 3.8444, correct 37, time 1.5399 seconds
Epoch 50: loss 4.0794, correct 45, time 1.5533 seconds
Epoch 60: loss 2.1984, correct 45, time 1.8597 seconds
Epoch 70: loss 3.3678, correct 46, time 1.5869 seconds
Epoch 80: loss 2.2343, correct 48, time 1.5439 seconds
Epoch 90: loss 3.7301, correct 46, time 1.6018 seconds
Epoch 100: loss 2.4227, correct 49, time 1.6133 seconds
Epoch 110: loss 2.4523, correct 48, time 2.0971 seconds
Epoch 120: loss 0.9952, correct 48, time 1.5379 seconds
Epoch 130: loss 2.0921, correct 49, time 1.5458 seconds
Epoch 140: loss 2.9144, correct 49, time 1.5973 seconds
Epoch 150: loss 1.1494, correct 49, time 1.5266 seconds
Epoch 160: loss 0.5875, correct 49, time 1.7492 seconds
Epoch 170: loss 1.6886, correct 50, time 1.5373 seconds
Epoch 180: loss 1.5263, correct 48, time 1.5412 seconds
Epoch 190: loss 1.6527, correct 48, time 2.2424 seconds
Epoch 200: loss 1.1253, correct 49, time 1.5359 seconds
Epoch 210: loss 0.4759, correct 49, time 1.5321 seconds
Epoch 220: loss 2.3503, correct 49, time 1.6116 seconds
Epoch 230: loss 1.0941, correct 49, time 1.5906 seconds
Epoch 240: loss 1.6934, correct 49, time 1.5323 seconds
Epoch 250: loss 0.9679, correct 49, time 1.6007 seconds
Epoch 260: loss 2.1667, correct 49, time 1.5543 seconds
Epoch 270: loss 0.4508, correct 49, time 2.1147 seconds
Epoch 280: loss 0.4729, correct 49, time 2.0886 seconds
Epoch 290: loss 2.4171, correct 49, time 1.5399 seconds
Epoch 300: loss 1.7243, correct 49, time 1.5334 seconds
Epoch 310: loss 0.2182, correct 50, time 1.5724 seconds
Epoch 320: loss 1.0518, correct 49, time 1.7097 seconds
Epoch 330: loss 0.7752, correct 49, time 1.5236 seconds
Epoch 340: loss 0.7625, correct 49, time 1.5702 seconds
Epoch 350: loss 1.2757, correct 49, time 2.1463 seconds
Epoch 360: loss 0.2715, correct 49, time 1.5230 seconds
Epoch 370: loss 0.3678, correct 49, time 1.5243 seconds
Epoch 380: loss 0.5980, correct 49, time 1.9063 seconds
Epoch 390: loss 0.9973, correct 48, time 1.6006 seconds
Epoch 400: loss 0.0579, correct 49, time 1.5267 seconds
Epoch 410: loss 2.2769, correct 49, time 1.5981 seconds
Epoch 420: loss 1.1255, correct 49, time 1.5380 seconds
Epoch 430: loss 0.1783, correct 49, time 1.6709 seconds
Epoch 440: loss 0.4582, correct 49, time 1.5312 seconds
Epoch 450: loss 1.4430, correct 49, time 1.5811 seconds
Epoch 460: loss 2.0488, correct 49, time 2.6142 seconds
Epoch 470: loss 0.9431, correct 49, time 1.6620 seconds
Epoch 480: loss 0.2549, correct 49, time 2.2684 seconds
Epoch 490: loss 1.6463, correct 49, time 1.5325 seconds
```
##### GPU
```bash
!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
```
```text
Epoch 0: loss 7.3287, correct 33, time 13.6268 seconds
Epoch 10: loss 5.1145, correct 45, time 0.1183 seconds
Epoch 20: loss 3.9054, correct 46, time 0.1086 seconds
Epoch 30: loss 2.5360, correct 50, time 0.1681 seconds
Epoch 40: loss 3.2986, correct 50, time 0.2267 seconds
Epoch 50: loss 2.2551, correct 50, time 0.1077 seconds
Epoch 60: loss 1.8457, correct 50, time 0.1076 seconds
Epoch 70: loss 2.2689, correct 49, time 0.1073 seconds
Epoch 80: loss 1.8842, correct 50, time 0.1087 seconds
Epoch 90: loss 1.8663, correct 50, time 0.1085 seconds
Epoch 100: loss 0.6110, correct 50, time 0.1096 seconds
Epoch 110: loss 0.3518, correct 50, time 0.1080 seconds
Epoch 120: loss 0.2245, correct 50, time 0.1080 seconds
Epoch 130: loss 0.8942, correct 50, time 0.1124 seconds
Epoch 140: loss 1.1848, correct 49, time 0.1852 seconds
Epoch 150: loss 0.6268, correct 50, time 0.1185 seconds
Epoch 160: loss 0.4116, correct 50, time 0.1068 seconds
Epoch 170: loss 1.0901, correct 50, time 0.1167 seconds
Epoch 180: loss 0.2280, correct 50, time 0.1073 seconds
Epoch 190: loss 0.8837, correct 50, time 0.1085 seconds
Epoch 200: loss 0.3247, correct 50, time 0.1097 seconds
Epoch 210: loss 0.4475, correct 50, time 0.1093 seconds
Epoch 220: loss 0.4252, correct 50, time 0.1074 seconds
Epoch 230: loss 0.1378, correct 50, time 0.1074 seconds
Epoch 240: loss 0.1358, correct 50, time 0.2058 seconds
Epoch 250: loss 0.1371, correct 50, time 0.1070 seconds
Epoch 260: loss 0.4791, correct 50, time 0.1093 seconds
Epoch 270: loss 0.1114, correct 50, time 0.1142 seconds
Epoch 280: loss 0.3445, correct 50, time 0.1205 seconds
Epoch 290: loss 0.1791, correct 50, time 0.1080 seconds
Epoch 300: loss 0.4925, correct 50, time 0.1076 seconds
Epoch 310: loss 0.1027, correct 50, time 0.1079 seconds
Epoch 320: loss 0.6227, correct 50, time 0.1068 seconds
Epoch 330: loss 0.0943, correct 50, time 0.1068 seconds
Epoch 340: loss 0.4371, correct 50, time 0.2399 seconds
Epoch 350: loss 0.2782, correct 50, time 0.1099 seconds
Epoch 360: loss 0.3965, correct 50, time 0.1057 seconds
Epoch 370: loss 0.2929, correct 50, time 0.1069 seconds
Epoch 380: loss 0.4778, correct 50, time 0.1129 seconds
Epoch 390: loss 0.7010, correct 50, time 0.1066 seconds
Epoch 400: loss 0.3196, correct 50, time 0.1071 seconds
Epoch 410: loss 0.0938, correct 50, time 0.1197 seconds
Epoch 420: loss 0.4754, correct 50, time 0.1064 seconds
Epoch 430: loss 0.0343, correct 50, time 0.1055 seconds
Epoch 440: loss 0.1931, correct 50, time 0.2460 seconds
Epoch 450: loss 0.3835, correct 50, time 0.1073 seconds
Epoch 460: loss 0.3498, correct 50, time 0.1067 seconds
Epoch 470: loss 0.0931, correct 50, time 0.1057 seconds
Epoch 480: loss 0.0475, correct 50, time 0.1085 seconds
Epoch 490: loss 0.0440, correct 50, time 0.1072 seconds
```

#### Simple
##### CPU
```bash
!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05
```
```text
Epoch 0: loss 4.7341, correct 37, time 13.6657 seconds
Epoch 10: loss 1.6585, correct 49, time 0.1087 seconds
Epoch 20: loss 1.5194, correct 49, time 0.1060 seconds
Epoch 30: loss 0.1533, correct 48, time 0.1065 seconds
Epoch 40: loss 1.7969, correct 48, time 0.1074 seconds
Epoch 50: loss 0.1513, correct 49, time 0.1062 seconds
Epoch 60: loss 0.4622, correct 49, time 0.1100 seconds
Epoch 70: loss 2.4976, correct 49, time 0.1183 seconds
Epoch 80: loss 1.3131, correct 49, time 0.1089 seconds
Epoch 90: loss 0.1510, correct 49, time 0.1952 seconds
Epoch 100: loss 1.4365, correct 49, time 0.2218 seconds
Epoch 110: loss 0.4516, correct 49, time 0.1057 seconds
Epoch 120: loss 0.4363, correct 49, time 0.1071 seconds
Epoch 130: loss 0.5580, correct 49, time 0.1086 seconds
Epoch 140: loss 2.6056, correct 48, time 0.1074 seconds
Epoch 150: loss 0.7725, correct 49, time 0.1079 seconds
Epoch 160: loss 0.4680, correct 49, time 0.1114 seconds
Epoch 170: loss 1.1815, correct 49, time 0.1067 seconds
Epoch 180: loss 1.2659, correct 49, time 0.1082 seconds
Epoch 190: loss 2.8119, correct 48, time 0.1301 seconds
Epoch 200: loss 0.1016, correct 49, time 0.1837 seconds
Epoch 210: loss 0.2216, correct 49, time 0.1074 seconds
Epoch 220: loss 0.0539, correct 50, time 0.1086 seconds
Epoch 230: loss 0.5330, correct 49, time 0.1207 seconds
Epoch 240: loss 0.1748, correct 49, time 0.1092 seconds
Epoch 250: loss 0.6681, correct 49, time 0.1082 seconds
Epoch 260: loss 0.9165, correct 48, time 0.1079 seconds
Epoch 270: loss 0.6127, correct 49, time 0.1069 seconds
Epoch 280: loss 0.1257, correct 49, time 0.1067 seconds
Epoch 290: loss 0.1491, correct 49, time 0.1047 seconds
Epoch 300: loss 0.0474, correct 49, time 0.1828 seconds
Epoch 310: loss 0.5322, correct 49, time 0.1140 seconds
Epoch 320: loss 0.4588, correct 50, time 0.1084 seconds
Epoch 330: loss 0.1358, correct 50, time 0.1060 seconds
Epoch 340: loss 0.8931, correct 50, time 0.1067 seconds
Epoch 350: loss 0.0744, correct 49, time 0.1098 seconds
Epoch 360: loss 0.9089, correct 50, time 0.1064 seconds
Epoch 370: loss 0.0003, correct 49, time 0.1060 seconds
Epoch 380: loss 0.0222, correct 50, time 0.1087 seconds
Epoch 390: loss 0.8153, correct 49, time 0.1071 seconds
Epoch 400: loss 0.3896, correct 49, time 0.2466 seconds
Epoch 410: loss 0.8832, correct 50, time 0.1094 seconds
Epoch 420: loss 1.3295, correct 50, time 0.1086 seconds
Epoch 430: loss 0.4143, correct 50, time 0.1075 seconds
Epoch 440: loss 0.3098, correct 49, time 0.1065 seconds
Epoch 450: loss 0.6615, correct 50, time 0.1076 seconds
Epoch 460: loss 0.4104, correct 49, time 0.1182 seconds
Epoch 470: loss 0.7184, correct 50, time 0.1061 seconds
Epoch 480: loss 0.1825, correct 49, time 0.1082 seconds
Epoch 490: loss 1.0286, correct 50, time 0.1060 seconds
```
##### GPU
```bash
!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05
```
```text
Epoch 0: loss 6.0708, correct 49, time 4.6543 seconds
Epoch 10: loss 2.2963, correct 49, time 1.6066 seconds
Epoch 20: loss 1.7486, correct 49, time 1.5604 seconds
Epoch 30: loss 1.2312, correct 49, time 1.9209 seconds
Epoch 40: loss 0.7836, correct 49, time 1.5474 seconds
Epoch 50: loss 0.7556, correct 48, time 1.5526 seconds
Epoch 60: loss 0.5748, correct 48, time 1.5512 seconds
Epoch 70: loss 1.4388, correct 49, time 1.6197 seconds
Epoch 80: loss 0.4033, correct 49, time 2.0452 seconds
Epoch 90: loss 0.5676, correct 50, time 1.6172 seconds
Epoch 100: loss 0.4638, correct 50, time 1.5740 seconds
Epoch 110: loss 0.0784, correct 49, time 1.7151 seconds
Epoch 120: loss 0.0873, correct 49, time 1.5628 seconds
Epoch 130: loss 0.1654, correct 50, time 1.8565 seconds
Epoch 140: loss 0.4859, correct 50, time 1.6463 seconds
Epoch 150: loss 0.0464, correct 50, time 1.5791 seconds
Epoch 160: loss 0.8585, correct 50, time 1.6330 seconds
Epoch 170: loss 0.9416, correct 49, time 1.6144 seconds
Epoch 180: loss 0.5037, correct 49, time 2.2915 seconds
Epoch 190: loss 0.3458, correct 50, time 1.5427 seconds
Epoch 200: loss 0.4620, correct 50, time 1.5367 seconds
Epoch 210: loss 0.2491, correct 50, time 1.6332 seconds
Epoch 220: loss 0.1995, correct 50, time 1.5543 seconds
Epoch 230: loss 0.1089, correct 50, time 2.0927 seconds
Epoch 240: loss 0.0393, correct 50, time 1.5582 seconds
Epoch 250: loss 0.0649, correct 50, time 1.6005 seconds
Epoch 260: loss 0.2872, correct 50, time 1.7993 seconds
Epoch 270: loss 1.2729, correct 49, time 1.5455 seconds
Epoch 280: loss 0.1949, correct 50, time 1.5414 seconds
Epoch 290: loss 0.0305, correct 50, time 1.5440 seconds
Epoch 300: loss 0.9198, correct 50, time 1.5963 seconds
Epoch 310: loss 0.5597, correct 50, time 2.1779 seconds
Epoch 320: loss 0.0388, correct 49, time 1.5921 seconds
Epoch 330: loss 0.1379, correct 50, time 1.5564 seconds
Epoch 340: loss 0.4610, correct 50, time 1.7113 seconds
Epoch 350: loss 0.0176, correct 50, time 1.5448 seconds
Epoch 360: loss 0.0379, correct 49, time 1.9014 seconds
Epoch 370: loss 0.4757, correct 50, time 1.5549 seconds
Epoch 380: loss 0.6328, correct 50, time 1.5553 seconds
Epoch 390: loss 0.0131, correct 50, time 1.9764 seconds
Epoch 400: loss 0.8702, correct 50, time 1.5411 seconds
Epoch 410: loss 0.3929, correct 50, time 1.6114 seconds
Epoch 420: loss 0.8671, correct 50, time 1.6326 seconds
Epoch 430: loss 0.7270, correct 49, time 1.5846 seconds
Epoch 440: loss 0.0362, correct 50, time 2.2743 seconds
Epoch 450: loss 0.6048, correct 50, time 1.5778 seconds
Epoch 460: loss 0.1461, correct 50, time 1.5680 seconds
Epoch 470: loss 0.3237, correct 50, time 1.5510 seconds
Epoch 480: loss 0.3762, correct 50, time 1.6495 seconds
Epoch 490: loss 0.2148, correct 50, time 1.7986 seconds
```

#### Xor 
##### CPU
```bash
!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05
```
```text
Epoch 0: loss 9.0808, correct 35, time 14.5282 seconds
Epoch 10: loss 4.3284, correct 43, time 0.1080 seconds
Epoch 20: loss 3.0612, correct 43, time 0.1086 seconds
Epoch 30: loss 3.9311, correct 43, time 0.1091 seconds
Epoch 40: loss 4.2223, correct 46, time 0.1115 seconds
Epoch 50: loss 2.0121, correct 46, time 0.1075 seconds
Epoch 60: loss 2.7100, correct 45, time 0.1101 seconds
Epoch 70: loss 3.5205, correct 46, time 0.1087 seconds
Epoch 80: loss 0.9341, correct 47, time 0.2654 seconds
Epoch 90: loss 1.9252, correct 47, time 0.1088 seconds
Epoch 100: loss 2.2958, correct 47, time 0.1096 seconds
Epoch 110: loss 1.2459, correct 47, time 0.1100 seconds
Epoch 120: loss 2.8326, correct 46, time 0.1212 seconds
Epoch 130: loss 1.7396, correct 46, time 0.1255 seconds
Epoch 140: loss 1.4541, correct 47, time 0.1209 seconds
Epoch 150: loss 1.3148, correct 49, time 0.1152 seconds
Epoch 160: loss 0.7506, correct 49, time 0.1136 seconds
Epoch 170: loss 2.0049, correct 47, time 0.1634 seconds
Epoch 180: loss 1.4792, correct 46, time 0.1332 seconds
Epoch 190: loss 1.6431, correct 50, time 0.1160 seconds
Epoch 200: loss 1.2466, correct 49, time 0.1076 seconds
Epoch 210: loss 1.2954, correct 46, time 0.1136 seconds
Epoch 220: loss 1.0771, correct 48, time 0.1167 seconds
Epoch 230: loss 1.7585, correct 47, time 0.1109 seconds
Epoch 240: loss 0.9973, correct 50, time 0.1116 seconds
Epoch 250: loss 2.5249, correct 49, time 0.1263 seconds
Epoch 260: loss 1.8650, correct 48, time 0.1254 seconds
Epoch 270: loss 2.2123, correct 48, time 0.2674 seconds
Epoch 280: loss 0.6920, correct 50, time 0.1098 seconds
Epoch 290: loss 0.3378, correct 50, time 0.1090 seconds
Epoch 300: loss 0.6113, correct 47, time 0.1101 seconds
Epoch 310: loss 0.9596, correct 50, time 0.1082 seconds
Epoch 320: loss 0.5095, correct 49, time 0.1075 seconds
Epoch 330: loss 1.4898, correct 48, time 0.1109 seconds
Epoch 340: loss 0.9676, correct 50, time 0.1202 seconds
Epoch 350: loss 0.4660, correct 50, time 0.1071 seconds
Epoch 360: loss 0.1806, correct 50, time 0.1145 seconds
Epoch 370: loss 0.3064, correct 49, time 0.2043 seconds
Epoch 380: loss 0.8557, correct 50, time 0.1070 seconds
Epoch 390: loss 1.2092, correct 50, time 0.1096 seconds
Epoch 400: loss 1.9625, correct 47, time 0.1075 seconds
Epoch 410: loss 0.7148, correct 49, time 0.1148 seconds
Epoch 420: loss 0.7597, correct 49, time 0.1075 seconds
Epoch 430: loss 0.2371, correct 50, time 0.1074 seconds
Epoch 440: loss 1.2876, correct 50, time 0.1081 seconds
Epoch 450: loss 0.3356, correct 50, time 0.1184 seconds
Epoch 460: loss 1.6292, correct 48, time 0.1138 seconds
Epoch 470: loss 0.3517, correct 50, time 0.1538 seconds
Epoch 480: loss 2.0742, correct 49, time 0.1083 seconds
Epoch 490: loss 1.5347, correct 49, time 0.1109 seconds
```
##### GPU
```bash
!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05
```
```text
Epoch 0: loss 8.2510, correct 32, time 3.9002 seconds
Epoch 10: loss 4.2599, correct 39, time 1.8269 seconds
Epoch 20: loss 3.7776, correct 41, time 1.5505 seconds
Epoch 30: loss 3.7547, correct 41, time 1.5609 seconds
Epoch 40: loss 5.7970, correct 40, time 1.5697 seconds
Epoch 50: loss 3.4388, correct 43, time 1.5523 seconds
Epoch 60: loss 2.9436, correct 45, time 1.9770 seconds
Epoch 70: loss 2.7605, correct 43, time 1.6197 seconds
Epoch 80: loss 2.5012, correct 42, time 1.5594 seconds
Epoch 90: loss 2.7055, correct 43, time 1.6109 seconds
Epoch 100: loss 1.9362, correct 44, time 1.5427 seconds
Epoch 110: loss 1.3870, correct 45, time 2.1178 seconds
Epoch 120: loss 3.1831, correct 45, time 1.5545 seconds
Epoch 130: loss 1.9423, correct 46, time 1.5609 seconds
Epoch 140: loss 1.9471, correct 45, time 1.7026 seconds
Epoch 150: loss 2.9726, correct 44, time 1.5960 seconds
Epoch 160: loss 2.2184, correct 45, time 1.8087 seconds
Epoch 170: loss 2.7900, correct 47, time 1.5577 seconds
Epoch 180: loss 3.3397, correct 48, time 1.5327 seconds
Epoch 190: loss 1.5639, correct 46, time 2.1475 seconds
Epoch 200: loss 2.6870, correct 47, time 1.5423 seconds
Epoch 210: loss 1.8245, correct 48, time 1.5414 seconds
Epoch 220: loss 1.4832, correct 49, time 1.5502 seconds
Epoch 230: loss 0.7483, correct 45, time 1.6042 seconds
Epoch 240: loss 2.3759, correct 48, time 2.2200 seconds
Epoch 250: loss 3.1945, correct 49, time 1.6293 seconds
Epoch 260: loss 2.7111, correct 49, time 1.5637 seconds
Epoch 270: loss 0.8248, correct 48, time 1.5430 seconds
Epoch 280: loss 1.7820, correct 49, time 1.5632 seconds
Epoch 290: loss 0.7569, correct 49, time 1.8242 seconds
Epoch 300: loss 1.0911, correct 49, time 1.5426 seconds
Epoch 310: loss 1.9853, correct 49, time 1.5478 seconds
Epoch 320: loss 2.1129, correct 50, time 2.0527 seconds
Epoch 330: loss 1.9943, correct 49, time 1.5465 seconds
Epoch 340: loss 0.8416, correct 49, time 1.6360 seconds
Epoch 350: loss 1.0505, correct 49, time 1.5769 seconds
Epoch 360: loss 0.9515, correct 49, time 1.5442 seconds
Epoch 370: loss 0.5017, correct 49, time 2.1795 seconds
Epoch 380: loss 0.5857, correct 49, time 1.5311 seconds
Epoch 390: loss 1.2953, correct 49, time 1.7121 seconds
Epoch 400: loss 0.6037, correct 50, time 1.5741 seconds
Epoch 410: loss 1.5894, correct 50, time 1.6127 seconds
Epoch 420: loss 1.7376, correct 50, time 1.8330 seconds
Epoch 430: loss 0.9242, correct 49, time 1.5372 seconds
Epoch 440: loss 1.2624, correct 50, time 1.5678 seconds
Epoch 450: loss 0.5323, correct 49, time 2.0976 seconds
Epoch 460: loss 0.6249, correct 49, time 1.5391 seconds
Epoch 470: loss 1.6218, correct 49, time 1.5825 seconds
Epoch 480: loss 0.9229, correct 49, time 1.5945 seconds
Epoch 490: loss 0.2734, correct 50, time 1.5820 seconds
```

### Bigger Model
#### Split
##### CPU
```bash
!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 200 --DATASET split --RATE 0.05
```
```text
Epoch 0: loss 8.3552, correct 33, time 13.8403 seconds
Epoch 10: loss 4.9601, correct 43, time 0.2350 seconds
Epoch 20: loss 2.3084, correct 48, time 0.2498 seconds
Epoch 30: loss 2.2669, correct 47, time 0.2427 seconds
Epoch 40: loss 1.5842, correct 49, time 0.2325 seconds
Epoch 50: loss 0.9593, correct 49, time 0.2513 seconds
Epoch 60: loss 0.7729, correct 49, time 0.5015 seconds
Epoch 70: loss 0.1610, correct 50, time 0.2348 seconds
Epoch 80: loss 1.0197, correct 49, time 0.2379 seconds
Epoch 90: loss 0.3279, correct 49, time 0.2355 seconds
Epoch 100: loss 0.3319, correct 50, time 0.2493 seconds
Epoch 110: loss 0.4785, correct 50, time 0.3609 seconds
Epoch 120: loss 0.3575, correct 49, time 0.2364 seconds
Epoch 130: loss 0.1344, correct 49, time 0.2480 seconds
Epoch 140: loss 0.4003, correct 50, time 0.2522 seconds
Epoch 150: loss 1.1274, correct 50, time 0.2381 seconds
Epoch 160: loss 0.4233, correct 49, time 0.2489 seconds
Epoch 170: loss 0.7817, correct 49, time 0.2394 seconds
Epoch 180: loss 0.0130, correct 50, time 0.2398 seconds
Epoch 190: loss 0.1840, correct 50, time 0.2443 seconds
Epoch 200: loss 0.8537, correct 49, time 0.4815 seconds
Epoch 210: loss 0.1102, correct 50, time 0.2491 seconds
Epoch 220: loss 0.1962, correct 50, time 0.2380 seconds
Epoch 230: loss 0.6839, correct 50, time 0.2344 seconds
Epoch 240: loss 0.0544, correct 50, time 0.2523 seconds
Epoch 250: loss 0.1315, correct 50, time 0.2377 seconds
Epoch 260: loss 0.1270, correct 50, time 0.2401 seconds
Epoch 270: loss 0.1015, correct 50, time 0.2335 seconds
Epoch 280: loss 0.0461, correct 50, time 0.2361 seconds
Epoch 290: loss 0.5068, correct 50, time 0.4886 seconds
Epoch 300: loss 0.1381, correct 50, time 0.2416 seconds
Epoch 310: loss 0.1346, correct 50, time 0.2396 seconds
Epoch 320: loss 0.4032, correct 50, time 0.2468 seconds
Epoch 330: loss 0.1998, correct 50, time 0.2380 seconds
Epoch 340: loss 0.0566, correct 50, time 0.2377 seconds
Epoch 350: loss 0.0971, correct 50, time 0.2379 seconds
Epoch 360: loss 0.1177, correct 50, time 0.2547 seconds
Epoch 370: loss 0.1139, correct 50, time 0.2527 seconds
Epoch 380: loss 0.0935, correct 50, time 0.2357 seconds
Epoch 390: loss 0.0807, correct 50, time 0.2355 seconds
Epoch 400: loss 0.3687, correct 50, time 0.2504 seconds
Epoch 410: loss 0.1745, correct 50, time 0.2370 seconds
Epoch 420: loss 0.3822, correct 50, time 0.2442 seconds
Epoch 430: loss 0.3406, correct 50, time 0.3917 seconds
Epoch 440: loss 0.2900, correct 50, time 0.2375 seconds
Epoch 450: loss 0.0414, correct 50, time 0.2483 seconds
Epoch 460: loss 0.1030, correct 50, time 0.2350 seconds
Epoch 470: loss 0.0188, correct 50, time 0.2352 seconds
Epoch 480: loss 0.1516, correct 50, time 0.2461 seconds
Epoch 490: loss 0.0251, correct 50, time 0.2362 seconds
```
##### GPU
```bash
!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 200 --DATASET split --RATE 0.05
```
```text
Epoch 0: loss 18.3632, correct 31, time 3.6759 seconds
Epoch 10: loss 3.9889, correct 41, time 1.6604 seconds
Epoch 20: loss 2.2669, correct 43, time 1.6173 seconds
Epoch 30: loss 3.5186, correct 46, time 1.6114 seconds
Epoch 40: loss 2.5992, correct 45, time 1.6293 seconds
Epoch 50: loss 1.7380, correct 49, time 1.9453 seconds
Epoch 60: loss 1.1405, correct 48, time 1.6488 seconds
Epoch 70: loss 1.4422, correct 46, time 2.2482 seconds
Epoch 80: loss 0.8530, correct 49, time 1.6266 seconds
Epoch 90: loss 1.3308, correct 47, time 1.6538 seconds
Epoch 100: loss 0.5553, correct 49, time 1.6326 seconds
Epoch 110: loss 1.2039, correct 49, time 1.6906 seconds
Epoch 120: loss 0.9478, correct 47, time 1.5984 seconds
Epoch 130: loss 0.4346, correct 49, time 1.5980 seconds
Epoch 140: loss 1.4892, correct 50, time 2.3818 seconds
Epoch 150: loss 1.2359, correct 49, time 1.6070 seconds
Epoch 160: loss 1.6588, correct 50, time 1.7161 seconds
Epoch 170: loss 0.9748, correct 47, time 1.6282 seconds
Epoch 180: loss 0.8107, correct 49, time 1.6964 seconds
Epoch 190: loss 0.2325, correct 49, time 1.6186 seconds
Epoch 200: loss 0.4263, correct 47, time 1.6699 seconds
Epoch 210: loss 0.4275, correct 49, time 1.7744 seconds
Epoch 220: loss 0.7024, correct 49, time 1.6225 seconds
Epoch 230: loss 0.9996, correct 48, time 2.4147 seconds
Epoch 240: loss 0.3681, correct 49, time 1.6032 seconds
Epoch 250: loss 1.4504, correct 47, time 1.6786 seconds
Epoch 260: loss 0.1622, correct 49, time 1.6095 seconds
Epoch 270: loss 0.9759, correct 49, time 1.6075 seconds
Epoch 280: loss 2.4381, correct 47, time 1.5946 seconds
Epoch 290: loss 1.4180, correct 49, time 1.6088 seconds
Epoch 300: loss 1.9450, correct 47, time 2.3617 seconds
Epoch 310: loss 0.4130, correct 49, time 1.6123 seconds
Epoch 320: loss 0.1156, correct 49, time 1.6584 seconds
Epoch 330: loss 0.3405, correct 49, time 1.5916 seconds
Epoch 340: loss 1.8889, correct 47, time 1.7009 seconds
Epoch 350: loss 0.5598, correct 49, time 1.6035 seconds
Epoch 360: loss 0.2845, correct 49, time 1.6351 seconds
Epoch 370: loss 0.7284, correct 49, time 2.3353 seconds
Epoch 380: loss 0.2039, correct 50, time 1.6160 seconds
Epoch 390: loss 0.7953, correct 49, time 1.6625 seconds
Epoch 400: loss 0.3781, correct 49, time 1.6077 seconds
Epoch 410: loss 0.2339, correct 48, time 1.6677 seconds
Epoch 420: loss 0.1258, correct 47, time 1.6922 seconds
Epoch 430: loss 0.0144, correct 49, time 1.6642 seconds
Epoch 440: loss 1.3748, correct 49, time 2.2655 seconds
Epoch 450: loss 1.8235, correct 49, time 1.6557 seconds
Epoch 460: loss 0.4205, correct 50, time 1.6289 seconds
Epoch 470: loss 1.9580, correct 47, time 1.5860 seconds
Epoch 480: loss 0.0142, correct 49, time 1.6615 seconds
Epoch 490: loss 0.0910, correct 48, time 1.5950 seconds
```

