# Astra
## A Math and Machine Learning Library Written In Rust
Status:
Conv2d net does not train well
missing a lot of tests (for tensor, for convolution and dense)
some decoupling needed in Tensor, Dense
need to add optimizers
need to add ability to save and load models

### Goals:
- [ ] Conv2d trained, with 90% accuracy by *05/10/2023*
- [ ] Tests
    - [ ] Tensor: transpose, qr, lu, reshape, slice, set slice, get column, set column, dot, norm, pad, rotate
    - [ ] Layer Dense
        - [ ] feed forward
        - [ ] back propagation
    - [ ] Layer Conv2d
        - [ ] Convolution (multiple strides, multiple padding etc..)
        - [ ] Calculate gradients per filter
        - [ ] calculate error to previous layer
        - [ ] feed forward
        - [ ] back propagation
    - [ ] Layer Flatten
        - [ ] feed forward
        - [ ] back propagation
    - [ ] loss functions
    - [ ] activation functions

- [ ] Decoupling
    - [ ] Layer Dense
        - [ ] gradients
        - [ ] parameters update
    - [ ] Layer Conv2d
        - [ ] move padding outside of convolution


- [ ] Optimizing
    - [ ] Layer Conv2d
        - [ ] calculate gradient per filter
        - [ ] calculate error for prev layer

### Longer Distance Goals:
- [ ] add macros
    - [ ] tensor 
    - [ ] layers
    - [ ] net
- [ ] Expanding
    - [ ] Statistical Algorithms 
    - [ ] Classic Machine Learning Algorithms
    - [ ] Genetic Algorithms
    - [ ] Swarm Intelligence Algorithms