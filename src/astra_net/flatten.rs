use crate::astra_net::layer::Layer;
use crate::error::AstraError;
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
    fn feed_forward(&mut self, inputs: &Tensor) -> Result<Tensor, AstraError> {
        self.input_shape = Some(inputs.shape.to_owned());
        inputs.to_owned().reshape(&[inputs.len()])
    }

    fn back_propagation(
        &mut self,
        error: Tensor,
        _learning_rate: f64,
        _clipping_value: Option<f64>,
    ) -> Result<Tensor, AstraError> {
        match self.input_shape.to_owned() {
            None => Err(AstraError::UninitializedLayerParameter(
                "self.input_shape".to_string(),
            )),
            Some(shape) => Ok(error.reshape(&shape)?),
        }
    }
}
