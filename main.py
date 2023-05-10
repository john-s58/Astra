# # This is a basic NN code to get a better hang on back propagation

import numpy as np


config = [3, 5, 3]

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
    return (y - target) * 2 / len(target)


class dense:
    def __init__(self, neurons, input_sz):
        self.weight = np.random.uniform(-1, 1, (input_sz, neurons))
        self.bias = np.full((1, neurons), 1.0)
    
    def ff(self, x):
        self.input = x
        self.output = np.dot(x, self.weight) + self.bias
        return relu(self.output)
    
    def bp(self, dl_dA):
        dout_dz = np.multiply(drelu(self.output), dl_dA)
        dz_dw = self.input

        gradient = np.dot(dz_dw.T, dout_dz)
        self.weight -= np.clip(gradient, -1, 1) * 0.01

        self.bias -= np.clip(np.sum(dout_dz, axis=0, keepdims=True), -1, 1) * 0.01

        dl_dx = np.dot(dout_dz, self.weight.T)

        return dl_dx

        
layers = []

for i in range(1, len(config)):
    layers.append(dense(config[i], config[i-1]))

ins = np.array([1, 1, 1])
ins = ins.reshape((1, len(ins)))
  
target = np.array([3, 3, 3])
target = target.reshape((1, len(target)))

res = ins
for l in layers:
    res = l.ff(res)
print(res)

for i in range(100):
    res = ins
    for l in layers:
        res = l.ff(res)

    loss = mse(res, target)
    dl_dout_layer = d_mse(res, target)

    for l in list(reversed(layers)):
        dl_dout_layer = l.bp(dl_dout_layer)


res = ins
for l in layers:
    res = l.ff(res)
print(res)

# res = np.array([2, 2, 2])
# res = res.reshape((1, len(res)))
# for l in layers:
#     res = l.ff(res)
# print(res)