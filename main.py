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
        self.bias = np.random.rand(neurons, 1)
    
    def ff(self, x):
        self.input = x
        self.output = self.weight.dot(x) + self.bias
        return relu(self.output)
    
    def bp(self, dl_dx):
        print("dl_dx shape", dl_dx.shape)

        dout_dz = drelu(self.output)

        dout_dz = dout_dz.reshape((dout_dz.size, 1))

        print("dout_dz shape", dout_dz.shape)

        dz_dw = self.input

        print("dz_dw shape", dz_dw.shape)

        gradient = dz_dw.T * dout_dz * dl_dx

        print("gradient shape", gradient.shape)

        print("weight shape", self.weight.shape)

        self.weight -= gradient * 0.001

        # dl_dx0 = (self
        # .weight.transpose()
        # .dot(drelu(self.output).reshape((1, len(self.output))))
        # .dot(dl_dx))

        return 1
        
layers = []

for i in range(1, len(config)):
    layers.append(dense(config[i], config[i-1]))

ins = np.array([1, 1, 1])
ins = ins.reshape((len(ins), 1))

res = ins
for l in layers:
    res = l.ff(res)
print(res)
  
target = np.array([3])

res = res.flatten()

print(f"res {res.shape}, target {target.shape}")

loss = mse(res, target)
dl_dout_layer = d_mse(res, target)
dl_dout_layer = dl_dout_layer.reshape((len(dl_dout_layer), 1))

for l in list(reversed(layers)):
    dl_dout_layer = l.bp(dl_dout_layer)
    break


res = input
for l in layers:
    res = l.ff(res)
print(res)