#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


def f(x):
    return 3 * x ** 2 - 4 * x + 5


# In[4]:


f(3)


# In[5]:


xs = np.arange(-5, 5, 0.25)
ys = f(xs)
plt.plot(xs, ys)


# In[6]:


def derivative(x, h = 0.000001):
    return (f(x + h) - f(x) )/ h


# In[7]:


derivative(1)


# In[8]:


h = 0.000001
a = 2
b = -3
c = 10

d1 = a * b + c
a += h 
d2 = a * b + c
print(d1)
print(d2)
print((d2 - d1) / h)


# In[2]:


import math


# In[1]:


class Value:
    def __init__(self, data, _children = (), _op = "", label = ""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda:None
        self._prev = set(_children)
        self._op = _op
        self.label = label
        
        
    def __add__(self, other):
        if isinstance(other, (int, float)):
            def _backward():
                self.grad += 1.0 * out.grad
            out = Value(self.data + other, _children = (self, ), _op = "+")
        elif isinstance(other, Value):
            out = Value(self.data + other.data, _children = (self, other), _op = "+")
            
            def _backward():
                other.grad += 1.0 * out.grad
                self.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            out = Value(self.data * other, _children = (self, ),_op = "*")            
            def _backward():
                self.grad += other * out.grad
            out = Value(self.data * other, _children = (self, ),_op = "*")
        elif isinstance(other, Value):
            out = Value(self.data * other.data, _children = (self, other), _op = "*")

            def _backward():
                other.grad += self.data * out.grad
                self.grad += other.data * out.grad
            out = Value(self.data * other.data, _children = (self, other), _op = "*")
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t =  (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1) 
        out = Value(t, _children = (self,), _op = "tanh")

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
            
        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), _children = (self, ), _op = "exp")
        def _backward():
            self.grad += x * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError("exponenta must be an integer or float: ")
        
        out = Value(math.pow(self.data, other), _children = (self, ), _op = "pow")
        def _backward():
            self.grad += other * math.pow(self.data, other - 1) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def topo_sort(d):
    
            if d not in visited:
                visited.add(d)
                for child_node in d._prev:
                    topo_sort(child_node)
                topo.append(d)
        topo_sort(self)   
        self.grad = 1
        for node in reversed(topo):
            node._backward()
            
            
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
        
    


# In[11]:


x1 = Value(2.0)
x2 = Value(0.0)
w1 = Value(-3.0)
w2 = Value(1)
b = Value(6.8813735870195432)
x1w1 = x1 * w1
x2w2 = x2 * w2
x1w1x2w2 = x1w1 + x2w2
n = x1w1x2w2 + b
o = n.tanh()
o.backward()


# In[12]:


print(x1)
print(w1)
print(x2)
print(w2)


# # # Same in PyTorch
# 

# In[13]:


import torch
x1 = torch.Tensor([2.0]); x1.requires_grad = True
x2 = torch.Tensor([0.0]); x2.requires_grad = True
w1 = torch.Tensor([-3.0]); w1.requires_grad = True
w2 = torch.Tensor([1.0]); w2.requires_grad = True
b = torch.Tensor([6.8813735870195432]); b.requires_grad = True


# In[14]:


n = x1 * w1 + x2 * w2 + b
o = torch.tanh(n)


# In[15]:


o.backward()


# In[16]:


print(x1, x1.grad.item())
print(w1, w1.grad.item())
print(x2, x2.grad.item())
print(w2, w2.grad.item())


# # NOW Let's implement MLP

# In[20]:


import math
import random

class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = np.zeros_like(self.data)  # Initialize grad with zeros_like data
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __add__(self, other):
        if isinstance(other, (int, float)):
            out = Value(self.data + other, _children=(self,), _op="+")
        elif isinstance(other, Value):
            out = Value(np.add(self.data, other.data), _children=(self, other), _op="+")

        def _backward():
            if isinstance(other, Value):
                other.grad += np.array(np.ones(out.data.shape)) * np.array(out.grad)
            self.grad += np.array(np.ones(out.data)) * np.array(out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            out = Value((self.data * other), _children=(self,), _op="*")
        elif isinstance(other, Value):
            out = Value(np.array(self.data) *np.array(other.data), _children=(self, other), _op="*")

        def _backward():
            if isinstance(other, Value):
                other.grad += np.array(self.data) * np.array(out.grad)
                self.grad += np.array(other.data) * np.array(out.grad)
            else:
                self.grad += np.array(other) * np.array(out.grad)
                

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, _children=(self,), _op="tanh")

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), _children=(self,), _op="exp")

        def _backward():
            self.grad += x * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError("exponent must be an integer or float: ")

        out = Value(math.pow(self.data, other), _children=(self,), _op="pow")

        def _backward():
            self.grad += other * math.pow(self.data, other - 1) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def topo_sort(d):
            if d not in visited:
                visited.add(d)
                for child_node in d._prev:
                    topo_sort(child_node)
                topo.append(d)

        topo_sort(self)
        self.grad = np.ones(self.data.shape)
        for node in reversed(topo):
            node._backward()

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other ** -1

    def __rtruediv__(self, other):  # other / self
        return other * self ** -1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


random.seed(42)


# In[237]:


a = Value([2, 3, 4])
b = Value([1, 2, 3])
c = a * b
c.backward()
b.grad


# In[107]:


np.random.seed(42)
class Neuron:
    def __init__(self, nin, nonlin=True):
        self.w = Value(np.random.uniform(-1,1, size = (1,10)))
       
    
    def __call__(self, x):
        act = np.dot(x , self.w)
        return act 
    
    def parameters(self):
        return self.w 
    
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y = True, as_frame = True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:





# In[118]:


random.seed(42)
import numpy as np
n = Neuron(X_train.shape[0])
# y_pred = [n(X_train.iloc[i,: ]) for i in range(X_train.shape[0])]
# y_pred
# X_train.shape[0]
# n(X_train.iloc[0,: ])
# y_pred = [n(X_train.iloc[i,:]) for i in range(X_train.shape[0])]
# n(X_train.iloc[9,:])
# n(X_train.iloc[11,:])
Value(np.random.uniform(-1,1, size = (10, 1))) * X_train.iloc[1, :]


# In[86]:


import tqdm
for k in tqdm.tqdm(range(100)):
#     forward pass
    y_pred = [n(X_train.iloc[:,i ]) for i in range(X_train.shape[0])]
    loss = sum((y_out - y_actual) ** 2 for y_actual, y_out in zip(y_train, y_pred))
    
#     zero_grad trick
    for p in n.parameters():
        p.grad = np.zeros_like(p)  # Initialize grad with zeros_like data
    
#     backward
    
    loss.backward()
    
#     update
    
    for p in n.parameters():
        p.data -= 0.1 * p.grad
#         print(p.data)
        
    print(k , loss.data)


# In[159]:


# MLP (Multilayer Perceptron) consists of 2 components: Layer and Neuron
# Neuron --->  Layer ---> MLP
import numpy as np
import torch
import random


# In[ ]:





# In[161]:


random.seed(42)
class Neuron:
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
    
    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act 
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer:
    def __init__(self, inp_number, output_number):
        self.neurons = [Neuron(inp_number) for _ in range(output_number)]
    def __call__(self,x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    def parameters(self):
        params = []
        for n in self.neurons:
            ps = n.parameters()
            params.extend(ps)
        return params
        
        
class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
#             for p in x:
#                 print(p)
#             print("--------------------------------------")
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

n = MLP(3, [4,4,1])



# In[164]:


# import tqdm
# for k in tqdm.tqdm(range(5)):
# #     forward pass
#     y_pred = [n(x) for x in xs]
#     loss = sum((y_out - y_actual) ** 2 for y_actual, y_out in zip(ys, y_pred))
    
# #     zero_grad trick
#     for p in n.parameters():
#         p.grad = 0.0
    
# #     backward
#     loss.backward()
    
# #     update
    
#     for p in n.parameters():
#         print(p)
#         p.data -= 0.1 * p.grad
# #         print(p.data)
        
# #     print(k , loss.data)
    
    


# In[22]:


y_pred


# # Scaling process in  hand

# In[101]:


# from sklearn.datasets import load_diabetes
# X, y = load_diabetes(return_X_y = True, as_frame = True, scaled = False)
# N, num_of_features = X.shape
# myu = np.mean(X, axis = 0)
# sigma = np.sqrt([(1/N) *sum((X.iloc[:, i] - myu[i]) ** 2)  for i in range(num_of_features)])
# A = ((X - myu) / (sigma * np.sqrt(N))).round(6)
# A


# In[154]:


from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y = True, as_frame = True)
X


# In[149]:


X_train = Value(X_train.values)
y_train = Value(y_train.values)


# In[155]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train


# In[152]:


import matplotlib.pyplot as plt
# for i in range(X_train.shape[1]):
#     plt.scatter(X_train.iloc[:, i] , y_train)
#     plt.show()


# In[6]:


n = Neuron(X_train.shape[0])


# In[163]:


import tqdm
for k in tqdm.tqdm(range(100)):
#     forward pass
    y_pred = [n(x) for x in X_train]
    loss = sum((y_out - y_actual) ** 2 for y_actual, y_out in zip(y_train, y_pred))
    
#     zero_grad trick
    for p in n.parameters():
        p.grad = 0.0
    
#     backward
    loss.backward()
    
#     update
    
    for p in n.parameters():
        p.data -= 0.1 * p.grad
#         print(p.data)
        
    print(k , loss.data)


# In[ ]:




