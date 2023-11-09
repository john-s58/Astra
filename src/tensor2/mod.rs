use crate::error::AstraError;
use crate::helper_traits::{Numeric, Float, Int, Unsigned};
use core::ops::{Add, Div, Mul, Sub, Rem};


pub struct Tensor<T> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
    pub n_dim: usize
} 

impl<T> Tensor<T> {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            shape: Vec::new(),
            n_dim: 0,
        }
    }

    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Result<Self, AstraError> {
        if shape.contains(&0) {
            return Err(AstraError::ZeroInShape);
        }
        if data.len() != shape.iter().product() {
            return Err(AstraError::ShapeDataMismatch);
        }

        Ok(Self {
            data,
            shape: shape.clone(),
            n_dim: shape.len(),
        })
    }

    pub fn from_element(element: T, shape: Vec<usize>) -> Result<Self, AstraError>
    where 
        T: Clone
     {
        if shape.contains(&0) {
            return Err(AstraError::ZeroInShape);
        }

        let data_size = shape.iter().product();
        Ok(Self {
            data: vec![element; data_size],
            shape: shape.clone(),
            n_dim: shape.len(),
        })
    }

    pub fn from_fn(shape: Vec<usize>, mut func: impl FnMut() -> T) -> Result<Self, AstraError> {
        if shape.contains(&0) {
            return Err(AstraError::ZeroInShape);
        }
        Ok(Self {
            data: (0..shape.iter().product()).map(|_| func()).collect(),
            shape: shape.clone(),
            n_dim: shape.len(),
        })
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn reshape(self, new_shape: &[usize]) -> Result<Self, AstraError> {
        if new_shape.contains(&0) {
            return Err(AstraError::ZeroInShape);
        }

        if self.shape.clone().into_iter().reduce(|a, b| a * b).unwrap()
            != new_shape.iter().copied().reduce(|a, b| a * b).unwrap()
        {
            return Err(AstraError::ShapeDataMismatch);
        }
        let n_dim = new_shape.len();
        Ok(Self {
            data: self.data,
            shape: Vec::from_iter(new_shape.iter().copied()),
            n_dim,
        })
    }

    pub fn get_index(&self, indices: &[usize]) -> Result<usize, AstraError> {
        let mut index = 0;
        let mut stride = 1;
        for i in 0..self.shape.len() {
            index += indices[i] * stride;
            stride *= self.shape[i];
        }
        if index > self.len() - 1 {
            return Err(AstraError::OutOfBounds);
        }
        Ok(index)
    }

    pub fn get_indices(&self, index: usize) -> Result<Vec<usize>, AstraError> {
        if index >= self.data.len() {
            return Err(AstraError::OutOfBounds);
        }

        let mut indices = vec![0; self.shape.len()];
        let mut remainder = index;

        for i in (0..self.shape.len()).rev() {
            let stride = self.shape[i];
            indices[i] = remainder % stride;
            remainder /= stride;
        }

        Ok(indices)
    }

    pub fn get_element(&self, indices: &[usize]) -> Result<&T, AstraError> {
        let index = self.get_index(indices)?;
        Ok(self.data.get(index).unwrap())
    }

    pub fn get_element_mut(&mut self, indices: &[usize]) -> Result<&mut T, AstraError> {
        let index = self.get_index(indices)?;
        Ok(self.data.get_mut(index).unwrap())
    }

    pub fn set_element(&mut self, indices: &[usize], value: T) -> Result<(), AstraError> {
        *self.get_element_mut(indices)? = value;
        Ok(())
    }
}

// impl<T: Numeric> Tensor<T>  {
//     pub fn dot(&self, other: &Self) -> Result<Self, AstraError> {
//         if self.shape.len() != 2 || other.shape.len() != 2 {
//             return Err(AstraError::UnsupportedDimension);
//         }
//         if self.shape[1] != other.shape[0] {
//             return Err(AstraError::ShapeMismatchBetweenTensors);
//         }
//         let mut result = Tensor::from_element(0.0, vec![self.shape[0], other.shape[1]])?;

//         for i in 0..self.shape[0] {
//             for j in 0..other.shape[1] {
//                 let mut val = 0.0;
//                 for k in 0..self.shape[1] {
//                     val += self.get_element(&[i, k])? * other.get_element(&[k, j])?;
//                 }
//                 *result.get_element_mut(&[i, j])? = val;
//             }
//         }

//         Ok(result)
//     }

// }


// impl<T> Tensor<T> where
//     T: Add<Output = Self>
//     + Div<Output = Self>
//     + Mul<Output = Self>
//     + Sub<Output = Self>
//     + Rem<Output = Self>
//     + Copy
//     + PartialEq
//     + PartialOrd
//     // Add the below if you don't want floats.
//     // + Eq
//     // + Ord
// {
//     pub fn identity(shape: Vec<usize>) -> Result<Self, AstraError> {
//         if !shape.windows(2).all(|w| w[0] == w[1]) {
//             return Err(AstraError::NonSquareMatrix);
//         }
//         let ndim = shape.len();
//         let elem_size = shape[0];
//         let mut res = Tensor::from_element(0.0, shape)?;
//         for i in 0..elem_size {
//             *(res.get_element_mut(&vec![i; ndim]))? = 1.0;
//         }
//         Ok(res)
//     }
// }