use std::error::Error;
use std::fmt;

#[derive(Debug, Clone)]
pub enum TensorError {
    ShapeDataMismatch,
    ZeroInShape,
    OutOfBounds,
    UnsupportedDimension,
    SingularMatrix,
    EmptyTensor,
    NonSquareMatrix,
    DimensionsMismatchForDotOperation,
    ShapeMismatchBetweenTensors,
    CustomError(String),
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::ZeroInShape => write!(f, "Shape contains 0"),
            TensorError::ShapeDataMismatch => write!(f, "Shape does not match data"),
            TensorError::OutOfBounds => write!(f, "Index out of bounds"),
            TensorError::UnsupportedDimension => write!(
                f,
                "Operation is unsupported for a Tensor with these dimensions"
            ),
            TensorError::EmptyTensor => write!(f, "Tensor is empty"),
            TensorError::SingularMatrix => write!(f, "Tensor is a singular matrix, non invertible"),
            TensorError::NonSquareMatrix => {
                write!(f, "Operation does not work on a non square matrix")
            }
            TensorError::DimensionsMismatchForDotOperation => {
                write!(f, "Left tensor n cols do not match right tensor n rows")
            }
            TensorError::ShapeMismatchBetweenTensors => {
                write!(f, "Shapes of right and left Tensors do not match")
            }
            TensorError::CustomError(s) => {
                write!(f, "Custom Error: {:?}", s)
            }
        }
    }
}

impl Error for TensorError {}
