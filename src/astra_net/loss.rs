use crate::tensor::tensor_error::TensorError;
use crate::tensor::Tensor;

use std::f64::EPSILON;

pub trait Loss {
    fn calculate(&self, output: &Tensor, target: &Tensor) -> Result<f64, TensorError>;
    fn get_output_layer_error(
        &self,
        output: &Tensor,
        target: &Tensor,
    ) -> Result<Tensor, TensorError>;
}

pub struct MSE;

impl MSE {
    pub fn new() -> Self {
        Self {}
    }
}

impl Loss for MSE {
    fn calculate(&self, output: &Tensor, target: &Tensor) -> Result<f64, TensorError> {
        if output.shape != target.shape {
            return Err(TensorError::ShapeMismatchBetweenTensors);
        }
        Ok(output
            .to_vec()
            .into_iter()
            .zip(target.to_vec().into_iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            / 3.)
    }
    fn get_output_layer_error(
        &self,
        output: &Tensor,
        target: &Tensor,
    ) -> Result<Tensor, TensorError> {
        if output.shape != target.shape {
            return Err(TensorError::ShapeMismatchBetweenTensors);
        }
        Ok(Tensor::from_vec(
            output
                .to_vec()
                .into_iter()
                .zip(target.to_vec().into_iter())
                .map(|(a, b)| ((a - b) * (a - b)) / 3.)
                .collect(),
            output.shape.to_owned(),
        )?)
    }
}

pub struct CategoricalCrossEntropy;

impl CategoricalCrossEntropy {
    pub fn new() -> Self {
        Self {}
    }
}

impl Loss for CategoricalCrossEntropy {
    fn calculate(&self, output: &Tensor, target: &Tensor) -> Result<f64, TensorError> {
        if output.shape != target.shape {
            return Err(TensorError::ShapeMismatchBetweenTensors);
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
    ) -> Result<Tensor, TensorError> {
        if output.shape != target.shape {
            return Err(TensorError::ShapeMismatchBetweenTensors);
        }
        Ok(Tensor::from_vec(
            output
                .to_vec()
                .into_iter()
                .zip(target.to_vec().into_iter())
                .map(|(a, b)| -(a + (b + EPSILON).ln()))
                .collect(),
            output.shape.to_owned(),
        )?)
    }
}
