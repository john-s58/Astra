# This is a basic NN code to get a better hang on back propagation

import numpy as np


config = [3, 5, 1]

def _lrelu(x):
    if x >= 0:
        return x
    return 0.3 * x

relu = np.vectorize(_lrelu)

def _dlrelru(x):
    if x >= 0:
        return 1
    return 0.3

drelu = np.vectorize(_dlrelru)


def mse(y, target):
   return sum((target - y) ** 2)

def d_mse(y, target):
    return (target - y) * 2 / len(target)


class dense:
    def __init__(self, neurons, input_sz):
        self.weight = np.random.rand(neurons, input_sz)
        self.bias = np.random.rand(neurons)
    
    def ff(self, x):
        self.input = x
        self.output = relu(self.weight.dot(x) + self.bias)
        return self.output
    
    def bp(self, dl_dx):
        #dout_dz = drelu(self.output)
        #dz_dw = self.input

        print("x shape: ", self.input.shape)
        print("out shape: ", self.output.shape)
        print("dl_dx_d_z shape: ", drelu(self.output).shape)
        print("dl_dx shape: ", dl_dx.shape)
        
        

        gradient = (self
        .input
        .reshape((len(self.input), 1))
        .dot(drelu(self.output)
        .reshape((len(self.output),1))).transpose()
        .dot(dl_dx))
    
        dl_dx0 = (self
        .weight.transpose()
        .dot(drelu(self.output).reshape((len(self.output),1)))
        .dot(dl_dx))

        self.weight -= gradient * 0.001
        return dl_dx0
        
layers = []

for i in range(1, len(config)):
    layers.append(dense(config[i], config[i-1]))

input = np.array([1, 1, 1])

res = input
for l in layers:
    res = l.ff(res)
print(res)
  
target = np.array([3])

loss = mse(res, target)
dl_dout_layer = d_mse(res, target)

for l in list(reversed(layers)):
    dl_dout_layer = l.bp(dl_dout_layer)

res = input
for l in layers:
    res = l.ff(res)
print(res)