pub mod activation;
pub mod conv2d;
pub mod conv2d2;
pub mod dense;
pub mod flatten;
pub mod layer;
pub mod loss;
pub mod net_error;

use crate::astra_net::layer::Layer;
use crate::astra_net::net_error::NetError;
use crate::tensor::Tensor;

pub struct Net {
    layers: Vec<Box<dyn Layer>>,
    learning_rate: f64,
}

impl Net {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            learning_rate: 0.001,
        }
    }

    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    pub fn feed_forward(&mut self, input: &Tensor) -> Result<Tensor, NetError> {
        let mut output = input.to_owned();
        for l in self.layers.iter_mut() {
            output = l.feed_forward(&output)?;
        }
        Ok(output)
    }

    pub fn back_propagation(&mut self, input: &Tensor, target: &Tensor) -> Result<(), NetError> {
        let output = self.feed_forward(input)?;

        let mut error = Tensor::from_vec(output.clone().to_vec(), vec![output.len()])
            .map_err(NetError::TensorBasedError)?;

        error = Tensor::from_vec(
            error
                .to_vec()
                .into_iter()
                .zip(target.to_owned().to_vec().into_iter())
                .map(|(x, y)| x - y)
                .collect(),
            vec![output.len()],
        )
        .map_err(NetError::TensorBasedError)?;

        for l in self.layers.iter_mut().rev() {
            error = l.back_propagation(error, 0.01)?;
        }
        Ok(())
    }
}
