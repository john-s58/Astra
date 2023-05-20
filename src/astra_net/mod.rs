pub mod activation;
pub mod conv2d2;
pub mod conv2d3;
pub mod dense;
pub mod flatten;
pub mod layer;
pub mod loss;

use crate::astra_net::layer::Layer;
use crate::error::AstraError;
use crate::tensor::Tensor;
use loss::Loss;

pub struct Net {
    layers: Vec<Box<dyn Layer>>,
    learning_rate: f64,
    loss: Box<dyn Loss>,
    clipping_value: Option<f64>,
}

impl Net {
    pub fn new(loss: Box<dyn Loss>, learning_rate: f64, clipping_value: Option<f64>) -> Self {
        Self {
            layers: Vec::new(),
            learning_rate,
            loss,
            clipping_value,
        }
    }

    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    pub fn feed_forward(&mut self, input: &Tensor) -> Result<Tensor, AstraError> {
        let mut output = input.to_owned();
        for l in self.layers.iter_mut() {
            output = l.feed_forward(&output)?;
        }
        Ok(output)
    }

    pub fn back_propagation(&mut self, input: &Tensor, target: &Tensor) -> Result<(), AstraError> {
        let output = self.feed_forward(input)?;

        let mut error = self
            .loss
            .get_output_layer_error(&output.clone().reshape(&[output.len()])?, target)?;

        for l in self.layers.iter_mut().rev() {
            error = l.back_propagation(error, self.learning_rate, self.clipping_value)?;
            // println!("ERR: {:?}", error);
            if error
                .clone()
                .into_iter()
                .any(|x| x.is_nan() || x.is_infinite())
            {
                panic!("err is nan or inf")
            }
        }
        Ok(())
    }
}
