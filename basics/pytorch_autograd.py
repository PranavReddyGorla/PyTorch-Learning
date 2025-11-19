import torch
x = torch.tensor(5.0, requires_grad=True)
y = 3*x**2 + 4
y.backward() # compute dy/dx
print(x.grad)  # prints 30.0 because dy/dx = 6x and at x=5, dy/dx=30




import torch
print("---------------------------------------------------------")
x = torch.tensor(2.0, requires_grad=True)
y = x**2
y.backward()
print(f"For x = {x.item()}, y = x^2 evaluates to y = {y.item()}")
print(f"The gradient dy/dx at x = {x.item()} is {x.grad.item()}")
print("---------------------------------------------------------")


x = torch.tensor(4.0, requires_grad=True)
y = x**2
z= torch.sin(y)
print(x)
print(y)
print(z)
z.backward()
print(x.grad)
print("---------------------------------------------------------")  



x = torch.tensor(6.7)
y = torch.tensor(0.0)
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
print(w)
print(b)
z = w*x + b
print(z)
y_pred = torch.sigmoid(z)
print(y_pred)
loss = (y_pred - y)**2 #define loss function
print(loss)
loss.backward() #dl/dw and dl/db
print(w.grad) 
print(b.grad)
print("---------------------------------------------------------")
