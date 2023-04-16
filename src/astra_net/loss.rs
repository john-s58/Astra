use crate::tensor::tensor_error::TensorError;
use crate::tensor::Tensor;

pub trait loss {
    fn calculate(&self, output: &Tensor, target: &Tensor) -> Result<f64, TensorError>;
    fn get_output_layer_error(
        &self,
        output: &Tensor,
        target: &Tensor,
    ) -> Result<Tensor, TensorError>;
}

pub struct MSE;

impl loss for MSE {
    fn calculate(&self, output: &Tensor, target: &Tensor) -> Result<f64, TensorError> {
        if output.shape != target.shape {
            return Err(TensorError::ShapeMismatchBetweenTensors);
        }
        Ok(output
            .to_vec()
            .into_iter()
            .zip(target.to_vec().into_iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum())
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
                .map(|(a, b)| (a - b) * (a - b))
                .collect(),
            output.shape.to_owned(),
        )?)
    }
}

pub struct CategoricalCrossEntropy;

impl loss for CategoricalCrossEntropy {
    fn calculate(&self, output: &Tensor, target: &Tensor) -> Result<f64, TensorError> {
        if output.shape != target.shape {
            return Err(TensorError::ShapeMismatchBetweenTensors);
        }
        todo!()
    }
    fn get_output_layer_error(
        &self,
        output: &Tensor,
        target: &Tensor,
    ) -> Result<Tensor, TensorError> {
        if output.shape != target.shape {
            return Err(TensorError::ShapeMismatchBetweenTensors);
        }
        todo!()
    }
}
