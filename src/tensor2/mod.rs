use crate::error::AstraError;
use crate::helper_traits::{Float, Int, Numeric, Unsigned};

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T> {
    pub shape: Vec<usize>,
    pub n_dim: usize,
    pub data: Vec<T>,
}

impl<T> Tensor<T>
where
    T: Numeric,
{
    pub fn new() -> Self {
        Self {
            n_dim: 0,
            shape: Vec::new(),
            data: Vec::new(),
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
            n_dim: shape.len(),
            shape,
            data,
        })
    }

    pub fn from_element(element: T, shape: Vec<usize>) -> Result<Self, AstraError> {
        if shape.contains(&0) {
            return Err(AstraError::ZeroInShape);
        }

        let data_size = shape.iter().product();
        Ok(Self {
            n_dim: shape.len(),
            shape,
            data: vec![element; data_size],
        })
    }

    pub fn from_fn(shape: Vec<usize>, mut func: impl FnMut() -> T) -> Result<Self, AstraError> {
        if shape.contains(&0) {
            return Err(AstraError::ZeroInShape);
        }
        Ok(Self {
            n_dim: shape.len(),
            shape: shape.clone(),
            data: (0..shape.iter().product()).map(|_| func()).collect(),
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
            n_dim,
            shape: new_shape.to_vec(),
            data: self.data,
        })
    }

    fn get_flat_index(indices: &[usize], shape: &Vec<usize>) -> Result<usize, AstraError> {
        let mut index = 0;
        let mut stride = 1;

        for (i, &dim_size) in shape.iter().enumerate().rev() {
            if indices[i] >= dim_size {
                return Err(AstraError::OutOfBounds);
            }
            index += indices[i] * stride;
            stride *= dim_size;
        }

        Ok(index)
    }

    pub fn get_index(&self, indices: &[usize]) -> Result<usize, AstraError> {
        if indices.len() != self.shape.len() {
            return Err(AstraError::ShapeDataMismatch);
        }
        let index = Self::get_flat_index(indices, &self.shape)?;
        Ok(index)
    }

    fn get_multi_index(index: usize, shape: &Vec<usize>) -> Vec<usize> {
        let mut indices = vec![0; shape.len()];
        let mut remainder = index;

        // Calculate strides for each dimension
        let mut strides = vec![1; shape.len()];
        for i in (1..shape.len()).rev() {
            strides[i - 1] = strides[i] * shape[i];
        }

        // Calculate indices for each dimension
        for (i, &stride) in strides.iter().enumerate() {
            indices[i] = remainder / stride;
            remainder %= stride;
        }
        indices
    }

    pub fn get_indices(&self, index: usize) -> Result<Vec<usize>, AstraError> {
        if index >= self.data.len() {
            return Err(AstraError::OutOfBounds);
        }
        let indices = Self::get_multi_index(index, &self.shape);
        Ok(indices)
    }

    pub fn get_element(&self, indices: &[usize]) -> Result<&T, AstraError> {
        let index = self.get_index(indices)?;
        Ok(self.data.get(index).unwrap()) // safe because of get_index
    }

    pub fn get_element_mut(&mut self, indices: &[usize]) -> Result<&mut T, AstraError> {
        let index = self.get_index(indices)?;
        Ok(self.data.get_mut(index).unwrap()) // safe because of get_index
    }

    pub fn set_element(&mut self, indices: &[usize], value: T) -> Result<(), AstraError> {
        *self.get_element_mut(indices)? = value;
        Ok(())
    }

    pub fn swap_element(
        &mut self,
        indices_a: &[usize],
        indices_b: &[usize],
    ) -> Result<(), AstraError> {
        let index_a = self.get_index(indices_a)?;
        let index_b = self.get_index(indices_b)?;
        self.data.swap(index_a, index_b);
        Ok(())
    }

    pub fn swap_rows(&mut self, row1: usize, row2: usize) -> Result<(), AstraError> {
        if self.n_dim != 2 {
            return Err(AstraError::UnsupportedDimension);
        }

        let row_count = self.shape[0];
        if row1 >= row_count || row2 >= row_count {
            return Err(AstraError::OutOfBounds);
        }

        let row_length = self.shape[1];
        for col in 0..row_length {
            self.swap_element(&[row1, col], &[row2, col])?;
        }

        Ok(())
    }

    pub fn swap_columns(&mut self, col1: usize, col2: usize) -> Result<(), AstraError> {
        if self.n_dim != 2 {
            return Err(AstraError::UnsupportedDimension);
        }

        let column_count = self.shape[1];
        if col1 >= column_count || col2 >= column_count {
            return Err(AstraError::OutOfBounds);
        }

        let row_length = self.shape[0];
        for row in 0..row_length {
            self.swap_element(&[row, col1], &[row, col2])?;
        }

        Ok(())
    }

    pub fn identity(shape: Vec<usize>) -> Result<Self, AstraError> {
        if shape.iter().any(|&dim| dim != shape[0]) {
            return Err(AstraError::NonSquareMatrix);
        }
        let size = shape[0];
        let total_elements = size * size;

        let mut data = vec![T::zero(); total_elements];
        for i in 0..size {
            data[i * size + i] = T::one();
        }

        Ok(Self {
            n_dim: 2,
            shape: vec![size, size],
            data,
        })
    }

    fn project_indices_to_ranges(indices: &[usize], ranges: &[(usize, usize)]) -> Vec<usize> {
        let mut projected: Vec<usize> = Vec::new();
        for (i, r) in indices.iter().zip(ranges.iter()) {
            projected.push(r.0 + i);
        }
        projected
    }

    pub fn slice(&self, ranges: &[(usize, usize)]) -> Result<Self, AstraError> {
        if self.n_dim != ranges.len() {
            return Err(AstraError::ShapeMismatchBetweenTensors);
        }

        // Ensure that ranges are within the bounds of the tensor dimensions
        for (&range, &dim_size) in ranges.iter().zip(self.shape.iter()) {
            if range.0 >= dim_size || range.1 >= dim_size || range.0 > range.1 {
                return Err(AstraError::OutOfBounds);
            }
        }

        let new_shape: Vec<usize> = ranges.iter().map(|&r| r.1 - r.0 + 1).collect();
        let new_size = new_shape.iter().product();
        let mut new_data = Vec::with_capacity(new_size);

        for flat_index in 0..new_size {
            let multi_index = Self::get_multi_index(flat_index, &new_shape);
            let projected_index = Self::project_indices_to_ranges(&multi_index, ranges);

            let value = self.get_element(&projected_index)?;
            new_data.push(*value);
        }
        Ok(Tensor::from_vec(new_data, new_shape)?)
    }

    pub fn set_slice(
        &mut self,
        ranges: &[(usize, usize)],
        source: &Self,
    ) -> Result<(), AstraError> {
        if self.n_dim != ranges.len() || self.n_dim != source.n_dim {
            return Err(AstraError::CustomError("Dimension mismatch".to_string()));
        }

        // Ensure ranges are within the bounds of the target tensor dimensions
        for (&range, &dim_size) in ranges.iter().zip(self.shape.iter()) {
            if range.0 >= dim_size || range.1 >= dim_size || range.0 > range.1 {
                return Err(AstraError::OutOfBounds);
            }
        }

        for dim in 0..self.n_dim {
            if ranges[dim].1 - ranges[dim].0 + 1 != source.shape[dim] {
                return Err(AstraError::CustomError(
                    format!(
                        "Slice shape mismatch {} != {}",
                        ranges[dim].1 - ranges[dim].0 + 1,
                        source.shape[dim]
                    )
                    .to_string(),
                ));
            }
        }

        for i in 0..source.data.len() {
            let source_multi_index = source.get_indices(i)?;
            let target_multi_index = Self::project_indices_to_ranges(&source_multi_index, ranges);

            let source_value = source.data[i];
            *self.get_element_mut(&target_multi_index)? = source_value;
        }

        Ok(())
    }

    pub fn transpose(&self) -> Result<Self, AstraError> {
        match self.n_dim {
            0 => Err(AstraError::EmptyTensor),
            1 => {
                // Transposing a 1D tensor is just reshaping it.
                Ok(Self::from_vec(self.data.clone(), vec![1, self.shape[0]]).unwrap())
            }
            2 => {
                let rows = self.shape[0];
                let cols = self.shape[1];

                if rows == cols {
                    // In-place transposition for square matrices
                    let mut data_clone = self.data.clone();
                    for i in 0..rows {
                        for j in i + 1..cols {
                            data_clone.swap(i * cols + j, j * rows + i);
                        }
                    }
                    Ok(Self {
                        data: data_clone,
                        shape: self.shape.clone(),
                        n_dim: 2,
                    })
                } else {
                    // Cache-friendly transposition for non-square matrices
                    let mut transposed_data = Vec::with_capacity(rows * cols);
                    for j in 0..cols {
                        for i in 0..rows {
                            transposed_data.push(self.data[i * cols + j].clone());
                        }
                    }
                    Ok(Self {
                        data: transposed_data,
                        shape: vec![cols, rows],
                        n_dim: 2,
                    })
                }
            }
            _ => Err(AstraError::UnsupportedDimension),
        }
    }

    pub fn dot(&self, other: &Self) -> Result<Self, AstraError> {
        // Check if self is a matrix
        if self.shape.len() != 2 {
            return Err(AstraError::UnsupportedDimension);
        }

        // Determine if other is a matrix or a vector
        let (other_cols, is_vector) = if other.shape.len() == 2 {
            // Check for matrix-matrix multiplication compatibility
            if self.shape[1] != other.shape[0] {
                return Err(AstraError::ShapeMismatchBetweenTensors);
            }
            (other.shape[1], false)
        } else if other.shape.len() == 1 {
            // Check for matrix-vector multiplication compatibility
            if self.shape[1] != other.shape[0] {
                return Err(AstraError::ShapeMismatchBetweenTensors);
            }
            (1, true)
        } else {
            return Err(AstraError::UnsupportedDimension);
        };

        let mut result_data = vec![T::default(); self.shape[0] * other_cols];
        let other_data = other.data.as_slice();

        for i in 0..self.shape[0] {
            for j in 0..other_cols {
                let mut val = T::default();
                for k in 0..self.shape[1] {
                    let self_index = i * self.shape[1] + k;
                    let other_index = if is_vector { k } else { k * other.shape[1] + j };
                    val = val + self.data[self_index] * other_data[other_index];
                }
                result_data[i * other_cols + j] = val;
            }
        }

        Ok(Tensor::from_vec(
            result_data,
            vec![self.shape[0], other_cols],
        )?)
    }

    pub fn gaussian_elimination(&mut self) -> Result<(), AstraError> {
        if self.n_dim != 2 {
            return Err(AstraError::UnsupportedDimension);
        }

        let rows = self.shape[0];
        let cols = self.shape[1];

        for i in 0..rows.min(cols) {
            println!("self in loop start {:#?}", self);
            // Find the pivot row
            let pivot_row = self.find_pivot_row(i)?;
            println!("pivot row: {}", pivot_row);

            // Swap pivot row with current row
            self.swap_rows(i, pivot_row)?;
            println!("self after swap_rows {:#?}", self);


            // Perform elimination
            self.eliminate_column(i)?;
            println!("self after eliminate column {:#?}", self);
        }

        Ok(())
    }

    pub fn find_pivot_row(&self, start_row: usize) -> Result<usize, AstraError> {
        let mut max = T::zero();
        let mut pivot_row = start_row;

        for r in start_row..self.shape[0] {
            let val = self.get_element(&[r, start_row])?;
            if val.abs() > max {
                max = val.abs();
                pivot_row = r;
            }
        }

        if max == T::zero() {
            return Err(AstraError::SingularMatrix);
        }

        Ok(pivot_row)
    }

    fn eliminate_column(&mut self, col: usize) -> Result<(), AstraError> {
        let pivot_val = *self.get_element(&[col, col])?;

        for r in col + 1..self.shape[0] {
            let factor = *self.get_element(&[r, col])? / pivot_val;
            for c in col..self.shape[1] {
                let val = *self.get_element(&[r, c])? - factor * *self.get_element(&[col, c])?;
                self.set_element(&[r, c], val)?;
            }
        }
        

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let tensor: Tensor<f32> = Tensor::new();
        assert!(tensor.data.is_empty());
        assert!(tensor.shape.is_empty());
        assert_eq!(tensor.n_dim, 0);
    }

    #[test]
    fn test_from_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let tensor = Tensor::from_vec(data.clone(), shape.clone()).unwrap();

        assert_eq!(tensor.data, data);
        assert_eq!(tensor.shape, shape);
        assert_eq!(tensor.n_dim, 2);
    }

    #[test]
    fn test_from_vec_with_invalid_shape() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![3, 2]; // Incorrect shape for the given data

        let result = Tensor::from_vec(data, shape);
        assert!(matches!(result, Err(AstraError::ShapeDataMismatch)));
    }

    #[test]
    fn test_from_element() {
        let tensor = Tensor::from_element(42, vec![2, 2]).unwrap();
        assert_eq!(tensor.data, vec![42, 42, 42, 42]);
        assert_eq!(tensor.shape, vec![2, 2]);
    }

    #[test]
    fn test_reshape() {
        let tensor = Tensor::from_element(1, vec![2, 3]).unwrap();
        let reshaped = tensor.reshape(&[3, 2]).unwrap();

        assert_eq!(reshaped.shape, vec![3, 2]);
        assert_eq!(reshaped.n_dim, 2);
    }

    #[test]
    fn test_reshape_with_invalid_shape() {
        let tensor = Tensor::from_element(1, vec![2, 3]).unwrap();
        let result = tensor.reshape(&[4, 1]);

        assert!(matches!(result, Err(AstraError::ShapeDataMismatch)));
    }
    #[test]
    fn test_get_index_and_element() {
        let tensor = Tensor::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let index = tensor.get_index(&[1, 1]).unwrap();
        let element = tensor.get_element(&[1, 1]).unwrap();

        assert_eq!(index, 3);
        assert_eq!(*element, 4);
    }

    #[test]
    fn test_get_indices() {
        let tensor = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let indices = tensor.get_indices(4).unwrap();

        assert_eq!(indices, vec![1, 1]);
    }

    #[test]
    fn test_set_element() {
        let mut tensor = Tensor::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        tensor.set_element(&[1, 1], 42).unwrap();

        assert_eq!(tensor.data, vec![1, 2, 3, 42]);
    }

    #[test]
    fn test_swap_rows() {
        // Create a 2x3 tensor
        let mut tensor = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();

        // Swap the two rows
        tensor.swap_rows(0, 1).unwrap();

        // Expected tensor after swap
        let expected = Tensor::from_vec(vec![4, 5, 6, 1, 2, 3], vec![2, 3]).unwrap();

        assert_eq!(tensor, expected);
    }

    #[test]
    fn test_swap_columns() {
        // Create a 3x2 tensor
        let mut tensor = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![3, 2]).unwrap();

        // Swap the two columns
        tensor.swap_columns(0, 1).unwrap();

        // Expected tensor after swap
        let expected = Tensor::from_vec(vec![2, 1, 4, 3, 6, 5], vec![3, 2]).unwrap();

        assert_eq!(tensor, expected);
    }

    #[test]
    fn test_identity() {
        let identity: Tensor<i64> = Tensor::identity(vec![3, 3]).unwrap();

        assert_eq!(identity.data, vec![1, 0, 0, 0, 1, 0, 0, 0, 1]);
        assert_eq!(identity.shape, vec![3, 3]);
    }

    #[test]
    fn test_transpose_square() {
        let tensor = Tensor::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let transposed = tensor.transpose().unwrap();

        assert_eq!(transposed.data, vec![1, 3, 2, 4]);
        assert_eq!(transposed.shape, vec![2, 2]);
    }

    #[test]
    fn test_transpose_non_square() {
        let tensor = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let transposed = tensor.transpose().unwrap();

        assert_eq!(transposed.data, vec![1, 4, 2, 5, 3, 6]);
        assert_eq!(transposed.shape, vec![3, 2]);
    }

    #[test]
    fn test_dot_matrix_matrix() {
        let a = Tensor::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5, 6, 7, 8], vec![2, 2]).unwrap();
        let result = a.dot(&b).unwrap();

        assert_eq!(result.data, vec![19, 22, 43, 50]);
    }

    #[test]
    fn test_dot_matrix_vector() {
        let a = Tensor::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5, 6], vec![2]).unwrap();
        let result = a.dot(&b).unwrap();

        assert_eq!(result.data, vec![17, 39]);
    }

    #[test]
    fn test_slice_normal() {
        let tensor = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], vec![3, 3]).unwrap();
        let sliced = tensor.slice(&[(1, 2), (0, 1)]).unwrap();
        let expected = Tensor::from_vec(vec![4, 5, 7, 8], vec![2, 2]).unwrap();
        assert_eq!(sliced, expected);
    }

    #[test]
    fn test_slice_single_element() {
        let tensor = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let sliced = tensor.slice(&[(0, 0), (2, 2)]).unwrap();
        let expected = Tensor::from_vec(vec![3], vec![1, 1]).unwrap();
        assert_eq!(sliced, expected);
    }

    #[test]
    fn test_slice_out_of_bounds() {
        let tensor = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let result = tensor.slice(&[(0, 2), (0, 3)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_slice_dimension_mismatch() {
        let tensor = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let result = tensor.slice(&[(0, 1)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_set_slice_normal() {
        let mut tensor = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let source = Tensor::from_vec(vec![7, 8], vec![1, 2]).unwrap();
        tensor.set_slice(&[(0, 0), (0, 1)], &source).unwrap();
        let expected = Tensor::from_vec(vec![7, 8, 3, 4, 5, 6], vec![2, 3]).unwrap();
        assert_eq!(tensor, expected);
    }

    #[test]
    fn test_set_slice_dimension_mismatch() {
        let mut tensor = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let source = Tensor::from_vec(vec![7, 8], vec![1, 2]).unwrap();
        let result = tensor.set_slice(&[(0, 1)], &source);
        assert!(result.is_err());
    }

    #[test]
    fn test_set_slice_out_of_bounds() {
        let mut tensor = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let source = Tensor::from_vec(vec![7, 8, 9, 10], vec![2, 2]).unwrap();
        let result = tensor.set_slice(&[(0, 2), (1, 3)], &source);
        assert!(result.is_err());
    }

    #[test]
    fn test_find_pivot_row_regular_matrix() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let pivot_row = tensor.find_pivot_row(0).unwrap();

        // Expect the pivot row to be the second row in this case
        assert_eq!(pivot_row, 1);
    }

    #[test]
    fn test_find_pivot_row_singular_matrix() {
        let tensor = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let result = tensor.find_pivot_row(0);

        assert!(matches!(result, Err(AstraError::SingularMatrix)));
    }

    #[test]
    fn test_eliminate_column_regular_matrix() {
        let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        tensor.eliminate_column(0).unwrap();

        // Expected result after eliminating the first column
        let expected = Tensor::from_vec(vec![1.0, 2.0, 0.0, -2.0], vec![2, 2]).unwrap();
        assert_eq!(tensor, expected);
    }

    #[test]
    fn test_eliminate_column_last_column() {
        let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = tensor.eliminate_column(1);

        // Expect success, but no changes as it's the last column
        assert!(result.is_ok());
        assert_eq!(
            tensor,
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap()
        );
    }

    #[test]
    fn test_gaussian_elimination_square_matrix() {
        let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        tensor.gaussian_elimination().unwrap();

        // Expected result after Gaussian Elimination
        let expected = Tensor::from_vec(vec![1.0, 2.0, 0.0, -2.0], vec![2, 2]).unwrap();
        assert_eq!(tensor, expected);
    }

    #[test]
    fn test_gaussian_elimination_rectangular_matrix() {
        let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        tensor.gaussian_elimination().unwrap();

        // Expected result after Gaussian Elimination
        let expected = Tensor::from_vec(vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0], vec![2, 3]).unwrap();
        assert_eq!(tensor, expected);
    }

    #[test]
    fn test_gaussian_elimination_unsupported_dimension() {
        let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let result = tensor.gaussian_elimination();

        assert!(matches!(result, Err(AstraError::UnsupportedDimension)));
    }

    #[test]
    fn test_gaussian_elimination_singular_matrix() {
        let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let result = tensor.gaussian_elimination();

        assert!(matches!(result, Err(AstraError::SingularMatrix)));
    }
}

// fn gaussian_elimination(&self) -> Result<(Self, i32), AstraError> {
//     if self.shape.len() != 2 {
//         return Err(AstraError::UnsupportedDimension);
//     }

//     let n = self.shape[0];
//     let mut matrix = self.data.clone();
//     let mut num_swaps = 0;

//     for i in 0..n {
//         // Partial pivoting
//         let mut max = i;
//         for k in i + 1..n {
//             if matrix[k * n + i].abs() > matrix[max * n + i].abs() {
//                 max = k;
//             }
//         }
//         if matrix[max * n + i] == T::default() {
//             // All remaining elements in column are zero, matrix might be singular
//             return Err(AstraError::SingularMatrix);
//         }
//         if i != max {
//             for j in 0..n {
//                 matrix.swap(i * n + j, max * n + j);
//             }
//             num_swaps += 1;
//         }

//         // Gaussian elimination
//         let pivot = matrix[i * n + i];
//         for k in i + 1..n {
//             let factor = matrix[k * n + i] / pivot;
//             for j in i..n {
//                 let value_to_subtract = factor * matrix[i * n + j];
//                 matrix[k * n + j] -= value_to_subtract;
//             }
//         }
//     }

//     // Constructing a new Tensor<T> with the modified matrix data
//     let tensor = Tensor::from_vec(matrix, self.shape.clone())?;
//     Ok((tensor, num_swaps))
// }

// pub fn det(&self) -> Result<T, AstraError> {
//     let (matrix, num_swaps) = self.gaussian_elimination()?;
//     let matrix = matrix.data.clone();
//     let mut det = if num_swaps % 2 == 0 {
//         T::from_i32(1)
//     } else {
//         T::from_i32(-1)
//     };

//     for i in 0..self.shape[0] {
//         det = det * matrix[i * self.shape[1] + i];
//     }

//     Ok(det)
// }
