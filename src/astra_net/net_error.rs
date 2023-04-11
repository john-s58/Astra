use crate::tensor::tensor_error::TensorError;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone)]
pub enum NetError {
    CustomError(String),
    TensorBasedError(TensorError),
    BadInputShape,
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
        }
    }
}

impl Error for NetError {}
