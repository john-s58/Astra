use crate::error::AstraError;
use crate::tensor::Tensor;

use std::f64::EPSILON;

pub trait Loss {
    fn calculate(&self, output: &Tensor, target: &Tensor) -> Result<f64, AstraError>;
    fn get_output_layer_error(
        &self,
        output: &Tensor,
        target: &Tensor,
    ) -> Result<Tensor, AstraError>;
}

pub struct MSE;

impl MSE {
    pub fn new() -> Self {
        Self {}
    }
}

impl Loss for MSE {
    fn calculate(&self, output: &Tensor, target: &Tensor) -> Result<f64, AstraError> {
        if output.shape != target.shape {
            return Err(AstraError::ShapeMismatchBetweenTensors);
        }
        Ok(output
            .to_vec()
            .into_iter()
            .zip(target.to_vec().into_iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            / target.len() as f64)
    }
    fn get_output_layer_error(
        &self,
        output: &Tensor,
        target: &Tensor,
    ) -> Result<Tensor, AstraError> {
        if output.shape != target.shape {
            return Err(AstraError::ShapeMismatchBetweenTensors);
        }
        Ok((output.to_owned() - target.to_owned()) * 2.0 / output.len() as f64)
    }
}

pub struct CategoricalCrossEntropy;

impl CategoricalCrossEntropy {
    pub fn new() -> Self {
        Self {}
    }
}

impl Loss for CategoricalCrossEntropy {
    fn calculate(&self, output: &Tensor, target: &Tensor) -> Result<f64, AstraError> {
        if output.shape != target.shape {
            return Err(AstraError::ShapeMismatchBetweenTensors);
        }
        Ok(output
            .to_vec()
            .into_iter()
            .zip(target.to_vec().into_iter())
            .map(|(a, b)| -(a + (b + EPSILON).ln()))
            .sum::<f64>())
    }
    fn get_output_layer_error(
        &self,
        output: &Tensor,
        target: &Tensor,
    ) -> Result<Tensor, AstraError> {
        if output.shape != target.shape {
            return Err(AstraError::ShapeMismatchBetweenTensors);
        }
        Ok((output.to_owned() - target.to_owned())
            / (output.to_owned() * output.to_owned().map(|x| 1.0 - x)))
    }
}
