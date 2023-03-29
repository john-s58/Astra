use crate::tensor::Tensor;

pub trait Layer {
    fn feed_forward(&mut self, inputs: &Tensor) -> Tensor;

    fn back_propagation(&mut self, error: Tensor, learning_rate: f64) -> Tensor;
}
