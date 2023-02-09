general idea
begin with a fully Dense neural network
transfrom the net structure to a vec of structs that implement the layer trait which contains feed_forward and back_propagation
activation function should be passed as a function parameter

## 04 Feb 2023
either find or create an array struct that can have any dimension and implements dot product, *, -, +, transpose etc
because some layers expect to recieve or return data that is not in the form of a single dimension array

### what's next:

next layer to implement will be the conv2d and flatten that comes with it

add activations sigmoid tanh

optimizers?

gradient clipping?

weight initialization of choice?

saving and loading models?

