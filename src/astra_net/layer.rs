use crate::astra_net::net_error::NetError;
use crate::tensor::Tensor;

pub trait Layer {
    fn feed_forward(&mut self, inputs: &Tensor) -> Result<Tensor, NetError>;

    fn back_propagation(&mut self, error: Tensor, learning_rate: f64) -> Result<Tensor, NetError>;
}
