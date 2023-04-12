use crate::astra_net::activation::Activation;
use crate::astra_net::layer::Layer;
use crate::astra_net::net_error::NetError;
use crate::tensor::Tensor;

pub struct LayerFlatten {
    input_shape: Option<Vec<usize>>,
}

impl LayerFlatten {
    pub fn new() -> Self {
        Self { input_shape: None }
    }
}

impl Layer for LayerFlatten {
    fn feed_forward(&mut self, inputs: &Tensor) -> Result<Tensor, NetError> {
        self.input_shape = Some(inputs.shape.to_owned());
        Ok(inputs
            .to_owned()
            .reshape(&[inputs.len()])
            .map_err(NetError::TensorBasedError)?)
    }

    fn back_propagation(&mut self, error: Tensor, learning_rate: f64) -> Result<Tensor, NetError> {
        match self.input_shape.to_owned() {
            None => Err(NetError::CustomError(
                "back prop called without feed forward on flat layer".to_string(),
            )),
            Some(shape) => Ok(error.reshape(&shape).map_err(NetError::TensorBasedError)?),
        }
    }
}
