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

weights = []
biases = []
zz = []
aa = []

for i in range(1, len(config)):
    weights.append(np.random.rand(config[i], config[i-1]))
    biases.append(np.random.rand(config[i]))

def ff(x):
    res = x
    for w,b in zip(weights, biases):
        res = w.dot(res) + b
        zz.append(res)
        aa.append(relu(res))
        
    return aa[len(aa) - 1]

def mse(y, target):
   return (target - y) ** 2

def d_mse(y, target):
    return (target - y) * 2 / len(target)

def bp(x, target):
    res = ff(x)
    err = d_mse(res, target)
    d_err = err

    for w, b in reversed(zip(weights, biases)):
        d_err = drelu(d_err)




print(ff(np.array([1,1,1])))

        