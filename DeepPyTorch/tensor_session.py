import torch
import numpy as np



# Creating a tensor from constructor
a = torch.FloatTensor(3,2)
print(a)
# The tensor created is unitiailized. It has random vallues. So we need to clear it
a.zero_()
print(a)

# Creating a tensor of diffrent dtypes from numpy arrays
#float64
n = np.random.random(10)
print(n)
print(torch.tensor(n, dtype=torch.float64))
#float32
print(torch.tensor(n, dtype=torch.float32))
#float16
print(torch.tensor(n, dtype=torch.float16))



# Scalar vectors. Demonstrationg `.item()`

a = torch.tensor([1,2,3])
print(a)
s = a.sum() # adds all the tensors values
print(s)
print(s.item())


#GPU and CPU

a = torch.FloatTensor([2,3])
print(a)
ca = a.cuda()
print(ca)