use crate::tensor::tensor_error::TensorError;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone)]
pub enum NetError {
    CustomError(String),
    TensorBasedError(TensorError),
    BadInputShape,
    UninitializedLayerParameter(String),
}

impl fmt::Display for NetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NetError::CustomError(s) => {
                write!(f, "Custom Error: {:?}", s)
            }
            NetError::TensorBasedError(e) => {
                write!(f, "Tensor source error {}", e)
            }
            NetError::BadInputShape => {
                write!(f, "Input does not match layer shape")
            }
            NetError::UninitializedLayerParameter(s) => {
                write!(f, "Trying to use uninitialized parameter {:?}", s)
            }
        }
    }
}

impl Error for NetError {}
