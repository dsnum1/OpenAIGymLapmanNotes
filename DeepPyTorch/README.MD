# Deep Learning with PyTorch

*PyTorch*
PyTorch is a tool that allows complex Deep Learning Models to be implemented in a few lines

*Tensor*
Tensor is a multi-dimensional array
Fundamental building block of all toolkits

*Creation of Tensor*
3 ways
    -   Calling constructor of the required type.
    -   Converting a numPy array or a python list into a tensor.
    -   Asking PyTorch to create a tensor with specific data for you.   torch.zeros()


`a.zero_()`
Reinitializes all elements inside tensor 'a' to 0.

*Operation for Tensors*
1. Inplace: 
    - Have an underscore appended
    - Operate on the tensor's content
    - Dangerous but more efficient
2. Functional 
    - Creates a copy of tensor with modifications performed. The original tensor is unchanged
    - Inefficient from a performance and memory point of view

For the kaggle challenge. Try to use as many inplace operators to decrease run time.

Usually in Deep Learning, double precision is not requireed and it adds extra memory and performance overheaad. So the common practicee is to use 32 bit float type or 16 bit float type. It's not needed to use the 64 bit float types.
