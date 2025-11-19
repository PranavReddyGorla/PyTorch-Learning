import torch
import numpy as np
import pandas as pd


scalar_tensor = torch.tensor(7)
vector_tensor = torch.tensor([1.0, 2.0, 3.0])
matrix_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("Scalar Tensor:", scalar_tensor)
print("Data Type of Scalar Tensor:", scalar_tensor.dtype)
print(scalar_tensor.device)
print("---------------------------------------------------------")
print("Vector Tensor:", vector_tensor)
print("Data Type of Vector Tensor:", vector_tensor.dtype)
print("---------------------------------------------------------")
print("Matrix Tensor:\n", matrix_tensor)
print("Data Type of Matrix Tensor:", matrix_tensor.dtype)
print("---------------------------------------------------------")


np_array = np.array([1, 2, 3, 4, 5])
tensor = torch.tensor(np_array)
print("NumPy Array:", np_array)
print("---------------------------------------------------------")

df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
tensor_from_df = torch.tensor(df.values)
print("Pandas DataFrame:\n", tensor_from_df)
print("---------------------------------------------------------")


random_tensor = torch.rand((3, 4))
print("Random Tensor:\n", random_tensor)
print("---------------------------------------------------------")

ones_tensor = torch.ones((2, 3))
print("Ones Tensor:\n", ones_tensor)
print("---------------------------------------------------------")

zeros_tensor = torch.zeros((2, 3))
print("Zeros Tensor:\n", zeros_tensor)
print("---------------------------------------------------------")

range_tensor = torch.arange(0, 10, step=2)
print("Range Tensor:", range_tensor)
print("---------------------------------------------------------")

linspace_tensor = torch.linspace(0, 1, steps=5)
print("Linspace Tensor:", linspace_tensor)
print("---------------------------------------------------------")


tensor_a = torch.tensor([[1, 2], [3, 4]])
tensor_b = torch.tensor([[5, 6], [7, 8]])
sum_tensor = tensor_a + tensor_b
print("Sum of Tensors:\n", sum_tensor)
diff_tensor = tensor_b - tensor_a
print("Difference of Tensors:\n", diff_tensor)
prod_tensor = tensor_a * tensor_b
print("Element-wise Product of Tensors:\n", prod_tensor)
matmul_tensor = torch.matmul(tensor_a, tensor_b)
print("Matrix Multiplication of Tensors:\n", matmul_tensor)
print("---------------------------------------------------------")


tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
numpy_array = tensor.numpy()
print("Tensor:\n", tensor)
print("Converted NumPy Array:\n", numpy_array)
print("---------------------------------------------------------")


tensor = torch.tensor([[1, 2], [3, 4]])
print(torch.sum(tensor))
print(torch.mean(tensor.float()))
print(torch.max(tensor))
print(torch.min(tensor))
print(torch.std(tensor.float()))
print("---------------------------------------------------------")

tensor = torch.tensor([1, 2, 3])
print("Original Tensor:", tensor)
print("Clipped Tensor:", tensor.clip(2, 3))
print("---------------------------------------------------------")

tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
reshaped_tensor = tensor.view(3, 2)
print("Original Tensor:\n", tensor)
print("Reshaped Tensor:\n", reshaped_tensor)
print("---------------------------------------------------------")

tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
transposed_tensor = tensor.t()
print("Original Tensor:\n", tensor)
print("Transposed Tensor:\n", transposed_tensor)
print("---------------------------------------------------------")


x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
dot_product = torch.dot(x, y)
print("Tensor x:", x)
print("Tensor y:", y)
print("Dot Product:", dot_product)
print("---------------------------------------------------------")


x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
result = torch.cat((x, y), dim=0)
print("Tensor x:", x)
print("Tensor y:", y)
print("Concatenated Tensor:", result)
print("---------------------------------------------------------")


x = torch.tensor([[1, 2], [3, 4]])
torch.save(x, 'tensor_x.pt')
loaded_x = torch.load('tensor_x.pt')
print("Original Tensor x:\n", x)
print("Loaded Tensor x from file:\n", loaded_x)
print("---------------------------------------------------------")