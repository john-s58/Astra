pub mod activation;
pub mod conv2d2;
pub mod dense;
pub mod flatten;
pub mod layer;
pub mod loss;
pub mod net_error;

use crate::astra_net::{layer::Layer, net_error::NetError};
use crate::tensor::Tensor;
use loss::Loss;

pub struct Net {
    layers: Vec<Box<dyn Layer>>,
    learning_rate: f64,
    loss: Box<dyn Loss>,
}

impl Net {
    pub fn new(loss: Box<dyn Loss>, learning_rate: f64) -> Self {
        Self {
            layers: Vec::new(),
            learning_rate,
            loss,
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

        let mut error = self
            .loss
            .get_output_layer_error(
                &output
                    .clone()
                    .reshape(&[output.len()])
                    .map_err(NetError::TensorBasedError)?,
                target,
            )
            .map_err(NetError::TensorBasedError)?;

        let mut i = self.layers.len() + 1;

        for l in self.layers.iter_mut().rev() {
            i -= 1;
            println!("ERR at layer {}: {:?}", i, error);
            error = l.back_propagation(error, 0.01)?;

            if error.to_vec().into_iter().any(|n| n.is_nan() || n > 15.0) {
                panic!("ERROR CONTAINS NAN")
            }
        }
        Ok(())
    }
}
