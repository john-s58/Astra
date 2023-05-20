use std::error::Error;
use std::fmt;

#[derive(Debug, Clone)]
pub enum AstraError {
    UnsupportedDimension,
    BadInputShape,
    OutOfBounds,
    EmptyTensor,
    SingularMatrix,
    ShapeMismatchBetweenTensors,
    NonSquareMatrix,
    ZeroInShape,
    ShapeDataMismatch,
    UninitializedLayerParameter(String),
    CustomError(String),
}

impl fmt::Display for AstraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AstraError::ZeroInShape => write!(f, "Shape contains 0"),
            AstraError::UnsupportedDimension => {
                write!(
                    f,
                    "Operation is unsupported for a Tensor with these dimensions"
                )
            }
            AstraError::BadInputShape => {
                write!(f, "Input shape does not match")
            }
            AstraError::OutOfBounds => {
                write!(f, "Index out of bounds")
            }
            AstraError::EmptyTensor => {
                write!(f, "Tensor is empty")
            }
            AstraError::SingularMatrix => {
                write!(f, "Operation is unsupported for a singular matrix")
            }
            AstraError::ShapeMismatchBetweenTensors => {
                write!(f, "Tensors shapes do not match")
            }
            AstraError::NonSquareMatrix => {
                write!(f, "Operation is unsupported for a non square matrix")
            }
            AstraError::ZeroInShape => {
                write!(f, "Tensor has 0 in shape")
            }
            AstraError::ShapeDataMismatch => {
                write!(f, "Tensor shape does not match data")
            }
            AstraError::UninitializedLayerParameter(s) => {
                write!(f, "Parameter {} is uninitialized", s)
            }
            AstraError::CustomError(s) => {
                write!(f, "Custom Error: {}", s)
            }
        }
    }
}

impl Error for AstraError {}
