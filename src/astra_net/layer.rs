use crate::error::AstraError;
use crate::tensor::Tensor;

pub trait Layer {
    fn feed_forward(&mut self, inputs: &Tensor) -> Result<Tensor, AstraError>;

    fn back_propagation(
        &mut self,
        output_gradient: Tensor,
        learning_rate: f64,
        clipping_value: Option<f64>,
    ) -> Result<Tensor, AstraError>;
}
