use crate::astra_net::activation::Activation;
use crate::astra_net::layer::Layer;
use crate::tensor::Tensor;

use ndarray_rand::rand_distr::{Distribution, Normal};

pub struct LayerConv2D {
    filters: Vec<Tensor>,
    stride: usize,
    padding: Option<Vec<usize>>,
    input_shape: Vec<usize>,
    activation: Box<dyn Activation>,
    input: Option<Tensor>,
    output: Option<Tensor>,
}

impl LayerConv2D {
    fn new() -> Self {
        todo!()
    }

    fn convolution(input: &Tensor, filter: &Tensor) {
        todo!()
    }
}

impl Layer for LayerConv2D {
    fn feed_forward(&mut self, inputs: &Tensor) -> Tensor{
        todo!()
    }

    fn back_propagation(&mut self, error: Tensor, learning_rate: f64) -> Tensor{
        todo!()
    }
}

