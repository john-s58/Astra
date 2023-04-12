pub mod tensor_error;

#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
    pub ndim: usize,
}

use std::env::temp_dir;
use std::f64;
use std::f64::EPSILON;

use self::tensor_error::TensorError;

// row first [3, 4] -> 3 rows 4 columns
// [2, 3, 4] -> 2 instances of 3 rows and 4 columns
impl Tensor {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            shape: Vec::new(),
            ndim: 0,
        }
    }

    pub fn from_vec(data: Vec<f64>, shape: Vec<usize>) -> Result<Self, TensorError> {
        if shape.contains(&0) {
            return Err(TensorError::ZeroInShape);
        }
        if data.len() != shape.iter().product() {
            return Err(TensorError::ShapeDataMismatch);
        }

        Ok(Self {
            data,
            shape: shape.clone(),
            ndim: shape.len(),
        })
    }

    pub fn from_element(element: f64, shape: Vec<usize>) -> Self {
        assert!(!shape.contains(&0), "shape contains 0");

        let data_size = shape.iter().product();
        Self {
            data: vec![element; data_size],
            shape: shape.clone(),
            ndim: shape.len(),
        }
    }

    pub fn from_fn(shape: Vec<usize>, mut func: impl FnMut() -> f64) -> Self {
        Self {
            data: (0..shape.iter().product()).map(|_| func()).collect(),
            shape: shape.clone(),
            ndim: shape.len(),
        }
    }

    pub fn reshape(self, new_shape: &[usize]) -> Result<Self, TensorError> {
        if new_shape.contains(&0) {
            return Err(TensorError::ZeroInShape);
        }

        if self.shape.clone().into_iter().reduce(|a, b| a * b).unwrap()
            != new_shape
                .to_owned()
                .into_iter()
                .reduce(|a, b| a * b)
                .unwrap()
        {
            return Err(TensorError::ShapeDataMismatch);
        }
        let ndim = new_shape.len();
        Ok(Self {
            data: self.data,
            shape: Vec::from_iter(new_shape.to_owned().into_iter()),
            ndim,
        })
    }

    pub fn identity(shape: Vec<usize>) -> Result<Self, TensorError> {
        if shape.windows(2).all(|w| w[0] == w[1]) == false {
            return Err(TensorError::NonSquareMatrix);
        }
        let ndim = shape.len();
        let elem_size = shape[0];
        let mut res = Tensor::from_element(0.0, shape);
        for i in 0..elem_size {
            *res.get_element_mut(&vec![i; ndim]).unwrap() = 1.0;
        }
        Ok(res)
    }

    pub fn matrix(m: usize, n: usize, data: Vec<f64>) -> Result<Self, TensorError> {
        match m * n == data.len() {
            false => Err(TensorError::ShapeDataMismatch),
            true => Ok(Self {
                data,
                shape: vec![m, n],
                ndim: 2,
            }),
        }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let ndim = shape.len();
        let size = shape.iter().product();

        Self {
            ndim,
            shape: shape.to_vec(),
            data: vec![0.0; size],
        }
    }

    pub fn transpose(&self) -> Result<Self, TensorError> {
        match self.ndim {
            0 => Err(TensorError::EmptyTensor),
            1 => Ok(self.to_owned().reshape(&[1, self.shape[0]])?),
            2 => {
                let mut transposed = self.to_owned();
                transposed.shape = transposed.shape.into_iter().rev().collect();
                for i in 0..self.shape[1] {
                    for j in 0..self.shape[0] {
                        *transposed.get_element_mut(&[i, j]).unwrap() =
                            self.get_element(&[j, i]).unwrap().to_owned();
                    }
                }
                Ok(transposed)
            }
            _ => Err(TensorError::UnsupportedDimension),
        }
    }

    pub fn get_index(&self, indices: &[usize]) -> Result<usize, TensorError> {
        let mut index = 0;
        let mut stride = 1;
        for i in 0..self.shape.len() {
            index += indices[i] * stride;
            stride *= self.shape[i];
        }
        if index > self.len() - 1 {
            return Err(TensorError::OutOfBounds);
        }
        Ok(index)
    }

    pub fn get_indices(&self, index: usize) -> Result<Vec<usize>, TensorError> {
        if index >= self.data.len() {
            return Err(TensorError::OutOfBounds);
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

    pub fn get_element(&self, indices: &[usize]) -> Result<&f64, TensorError> {
        let index = self.get_index(indices)?;
        Ok(&self.data.get(index).unwrap())
    }

    pub fn get_element_mut(&mut self, indices: &[usize]) -> Result<&mut f64, TensorError> {
        let index = self.get_index(indices)?;
        Ok(self.data.get_mut(index).unwrap())
    }

    pub fn set_element(&mut self, indices: &[usize], value: f64) -> Result<(), TensorError> {
        *self.get_element_mut(indices)? = value;
        Ok(())
    }

    pub fn map(self, fun: impl Fn(f64) -> f64) -> Self {
        Self {
            data: self.data.into_iter().map(fun).collect(),
            shape: self.shape,
            ndim: self.ndim,
        }
    }

    pub fn sum(&self) -> f64 {
        if self.data.len() == 1 {
            return self.data[0];
        }
        self.data.clone().into_iter().sum()
    }

    pub fn dot(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(TensorError::UnsupportedDimension);
        }
        if self.shape[1] != other.shape[0] {
            return Err(TensorError::DimensionsMismatchForDotOperation);
        }
        let mut result = Tensor::from_element(0.0, vec![self.shape[0], other.shape[1]]);

        for i in 0..self.shape[0] {
            for j in 0..other.shape[1] {
                let mut val = 0.0;
                for k in 0..self.shape[1] {
                    val += self.get_element(&[i, k])? * other.get_element(&[k, j])?;
                }
                *result.get_element_mut(&[i, j])? = val;
            }
        }

        Ok(result)
    }

    pub fn lu_decomposition(&self) -> Result<(Self, Self), TensorError> {
        if self.ndim != 2 {
            return Err(TensorError::UnsupportedDimension);
        }
        if self.shape[0] != self.shape[1] {
            return Err(TensorError::NonSquareMatrix);
        }

        let n = self.shape[0];
        let mut l = Tensor::zeros(&[n, n]);
        let mut u = Tensor::zeros(&[n, n]);

        for i in 0..n {
            for j in i..n {
                let mut sum = 0.0;
                for k in 0..i {
                    sum += *l.get_element(&[i, k]).unwrap() * *u.get_element(&[k, j]).unwrap();
                }

                let u_value = *self.get_element(&[i, j]).unwrap() - sum;
                *u.get_element_mut(&[i, j]).unwrap() = u_value;
            }

            for j in i..n {
                if i == j {
                    *l.get_element_mut(&[i, i]).unwrap() = 1.0;
                } else {
                    let mut sum = 0.0;
                    for k in 0..i {
                        sum += *l.get_element(&[j, k]).unwrap() * *u.get_element(&[k, i]).unwrap();
                    }

                    if u.get_element(&[i, i]).unwrap().abs() < EPSILON {
                        return Err(TensorError::CustomError(
                            format!("LU Decomposition failed due element at indices [{:?}, {:?}] >= EPSILON", i, i)
                            .to_string()));
                    }

                    let l_value = (*self.get_element(&[j, i]).unwrap() - sum)
                        / *u.get_element(&[i, i]).unwrap();
                    *l.get_element_mut(&[j, i]).unwrap() = l_value;
                }
            }
        }
        Ok((l, u))
    }

    pub fn norm(&self) -> Result<f64, TensorError> {
        match self.ndim {
            1 => {
                let mut sum = 0.0;
                for val in &self.data {
                    sum += val * val;
                }
                Ok(sum.sqrt())
            }
            2 => {
                let rows = self.shape[0];
                let cols = self.shape[1];
                let mut sum = 0.0;
                for i in 0..rows {
                    for j in 0..cols {
                        let val = self.data[i * cols + j];
                        sum += val * val;
                    }
                }
                Ok(sum.sqrt())
            }
            _ => Err(TensorError::UnsupportedDimension),
        }
    }

    pub fn qr(&self) -> Result<(Tensor, Tensor), TensorError> {
        if self.ndim != 2 {
            return Err(TensorError::UnsupportedDimension);
        }

        let rows = self.shape[0];
        let cols = self.shape[1];

        let mut q = Self::zeros(&[rows, cols]);
        let mut r = Self::zeros(&[cols, cols]);

        for k in 0..cols {
            let mut u_k = self.get_column(k);
            for i in 0..k {
                let q_i = q.get_column(i);
                let r_ik = q_i.dot(&u_k).unwrap().data[0];

                r.set_element(&[i, k], r_ik)?;
                u_k = u_k / q_i * r_ik;
            }
            let norm = u_k.norm()?;
            r.set_element(&[k, k], norm)?;
            q.set_column(k, &(u_k / norm));
        }

        Ok((q, r))
    }

    fn get_column(&self, col: usize) -> Self {
        let rows = self.shape[0];
        let mut data = Vec::with_capacity(rows);
        for i in 0..rows {
            data.push(self.data[i * self.shape[1] + col]);
        }
        Tensor {
            ndim: 1,
            shape: vec![rows],
            data,
        }
    }

    fn set_column(&mut self, col: usize, tensor: &Self) {
        let rows = self.shape[0];
        for i in 0..rows {
            self.data[i * self.shape[1] + col] = tensor.data[i];
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn to_vec(&self) -> Vec<f64> {
        self.data.to_owned()
    }

    pub fn get_sub_matrix(
        &self,
        left_corner: &[usize],
        shape: &[usize],
    ) -> Result<Self, TensorError> {
        if self.ndim != 2 {
            return Err(TensorError::UnsupportedDimension);
        }
        if left_corner.len() != 2 || shape.len() != 2 {
            return Err(TensorError::UnsupportedDimension);
        }

        let (left_corner_i, left_corner_j) = (left_corner[0], left_corner[1]);
        let (shape_i, shape_j) = (shape[0], shape[1]);

        if left_corner_i + shape_i > self.shape[0] || left_corner_j + shape_j > self.shape[1] {
            return Err(TensorError::CustomError(
                "sub matrix dimensions larger than original matrix".to_string(),
            ));
        }

        // Create a new tensor to store the sub-matrix
        let mut sub_matrix = Tensor::from_element(0.0, vec![shape_i, shape_j]);

        // Copy the data from the original tensor to the sub-matrix
        for i in 0..shape_i {
            for j in 0..shape_j {
                let value = *self
                    .get_element(&[left_corner_i + i, left_corner_j + j])
                    .unwrap();
                *sub_matrix.get_element_mut(&[i, j]).unwrap() = value;
            }
        }

        Ok(sub_matrix)
    }
    pub fn set_slice(
        &mut self,
        ranges: &[(usize, usize)],
        source: &Self,
    ) -> Result<(), TensorError> {
        if self.ndim < source.ndim || self.ndim != ranges.len() {
            return Err(TensorError::CustomError(
                "dimensions error with self, source and ranges".to_string(),
            ));
        }

        let target_dims: Vec<usize> = ranges
            .iter()
            .enumerate()
            .filter(|(_, r)| r.1 > r.0)
            .map(|(i, _)| i)
            .collect();

        if target_dims.len() != source.ndim {
            return Err(TensorError::CustomError(
                "source dimensions mismatch with non-single-value ranges".to_string(),
            ));
        }

        for (dim, &target_dim) in target_dims.iter().enumerate() {
            if (ranges[target_dim].1 - ranges[target_dim].0) + 1 != source.shape[dim] {
                return Err(TensorError::CustomError(
                    "slice shape different from source shape".to_string(),
                ));
            }
        }

        let mut index: Vec<usize> = ranges.iter().map(|r| r.0).collect();
        let mut src_index = vec![0; source.ndim];

        loop {
            // Copy element from the source tensor to the current tensor
            let value = *source.get_element(&src_index).unwrap();
            let target_index: Vec<usize> = index
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    if target_dims.contains(&i) {
                        *v + src_index[target_dims.iter().position(|&td| td == i).unwrap()]
                    } else {
                        *v
                    }
                })
                .collect();
            *self.get_element_mut(&target_index).unwrap() = value;

            // Increment the indices for the source tensor
            let mut dim = source.ndim - 1;
            while dim < source.ndim {
                src_index[dim] += 1;

                // Check if the index is within the specified range
                if src_index[dim] < source.shape[dim] {
                    break;
                } else {
                    // Reset the index for the current dimension
                    src_index[dim] = 0;

                    // Move to the previous dimension
                    if dim > 0 {
                        dim -= 1;
                    } else {
                        return Ok(()); // All elements processed, exit the loop
                    }
                }
            }
        }
    }

    pub fn slice(&self, ranges: &[(usize, usize)]) -> Result<Self, TensorError> {
        if self.ndim != ranges.len() {
            return Err(TensorError::ShapeMismatchBetweenTensors);
        }

        let new_shape: Vec<usize> = ranges.iter().map(|r| r.1 - r.0 + 1).collect();
        let new_size = new_shape.iter().product();

        let mut new_data = Vec::with_capacity(new_size);
        let mut index: Vec<usize> = ranges.iter().map(|r| r.0).collect();

        loop {
            // Copy element from the current tensor to the new tensor
            let value = *self.get_element(&index).unwrap();
            new_data.push(value);

            // Increment the indices for the current tensor
            let mut dim = self.ndim - 1;
            while dim < self.ndim {
                index[dim] += 1;

                // Check if the index is within the specified range
                if index[dim] <= ranges[dim].1 {
                    break;
                } else {
                    // Reset the index for the current dimension
                    index[dim] = ranges[dim].0;

                    // Move to the previous dimension
                    if dim > 0 {
                        dim -= 1;
                    } else {
                        return Ok(Tensor::from_vec(new_data, new_shape)?); // All elements processed, exit the loop
                    }
                }
            }
        }
    }

    pub fn pad(&self, padding: &[(usize, usize)]) -> Result<Self, TensorError> {
        if self.ndim != padding.len() {
            return Err(TensorError::ShapeMismatchBetweenTensors);
        }
        let new_shape: Vec<usize> = self
            .shape
            .iter()
            .zip(padding.iter())
            .map(|(dim_size, (pad_before, pad_after))| dim_size + pad_before + pad_after)
            .collect();
        let mut padded = Tensor::from_element(0.0, new_shape);

        let index_ranges: Vec<(usize, usize)> = padded
            .shape
            .to_owned()
            .into_iter()
            .zip(padding.into_iter())
            .map(|(dim_size, (pad_before, pad_after))| (*pad_before, dim_size - pad_after - 1))
            .collect();

        padded.set_slice(&index_ranges.as_slice(), &self)?;

        Ok(padded)
    }

    pub fn print_matrix(&self) -> Result<(), TensorError> {
        if self.ndim != 2 {
            return Err(TensorError::UnsupportedDimension);
        }

        let mut index = 0;
        for _row in 0..self.shape[0] {
            for _col in 0..self.shape[1] {
                print!("{:5.1} ", self.data[index]);
                index += 1;
            }
            println!();
        }
        Ok(())
    }

    pub fn stack(tensors: &[Self]) -> Result<Self, TensorError> {
        let base_shape = tensors[0].shape.clone();
        if tensors.iter().any(|t| t.shape != base_shape) {
            return Err(TensorError::ShapeMismatchBetweenTensors);
        }
        let n_tensors = tensors.len();

        let range_vector: Vec<(usize, usize)> =
            base_shape.clone().into_iter().map(|x| (0, x - 1)).collect();

        let mut stack_shape = base_shape.clone();
        stack_shape.insert(0, n_tensors);

        let mut stacked = Tensor::zeros(&stack_shape);

        for (i, t) in tensors.iter().enumerate() {
            let mut ranges = range_vector.clone();
            ranges.insert(0, (i, i));

            stacked.set_slice(&ranges, t)?;
        }
        Ok(stacked)
    }

    pub fn rotate_90_degrees(&self) -> Result<Self, TensorError> {
        if self.ndim != 2 {
            return Err(TensorError::UnsupportedDimension);
        }

        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut rotated = Tensor::from_element(0.0, vec![cols, rows]);

        for i in 0..rows {
            for j in 0..cols {
                let value = *self.get_element(&[i, j]).unwrap();
                *rotated.get_element_mut(&[j, rows - 1 - i]).unwrap() = value;
            }
        }

        Ok(rotated)
    }

    pub fn rotate_180_degrees(&self) -> Result<Self, TensorError> {
        if self.ndim != 2 {
            return Err(TensorError::UnsupportedDimension);
        }

        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut rotated = Tensor::from_element(0.0, vec![rows, cols]);

        for i in 0..rows {
            for j in 0..cols {
                let value = *self.get_element(&[i, j]).unwrap();
                *rotated
                    .get_element_mut(&[rows - 1 - i, cols - 1 - j])
                    .unwrap() = value;
            }
        }

        Ok(rotated)
    }
}

impl std::ops::Mul<Tensor> for f64 {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        Tensor {
            data: rhs.data.into_iter().map(|x| self * x).collect(),
            shape: rhs.shape,
            ndim: rhs.ndim,
        }
    }
}

impl std::ops::Mul<f64> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Self::Output {
        Tensor {
            data: self.data.into_iter().map(|x| rhs * x).collect(),
            shape: self.shape,
            ndim: self.ndim,
        }
    }
}

impl std::ops::Div<f64> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: f64) -> Self::Output {
        Tensor {
            data: self.data.into_iter().map(|x| x / rhs).collect(),
            shape: self.shape,
            ndim: self.ndim,
        }
    }
}

impl std::ops::Mul<Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        assert!(self.ndim == rhs.ndim, "Tensors DIMS do not match");
        for (idx, (dim_t1, dim_t2)) in self.shape.iter().zip(rhs.shape.iter()).enumerate() {
            assert!(
                dim_t1 == dim_t2,
                "Dimensions do not match at dim number {:?}",
                idx
            );
        }
        Tensor {
            data: self
                .data
                .into_iter()
                .zip(rhs.data.into_iter())
                .map(|(x, y)| x * y)
                .collect(),
            shape: self.shape,
            ndim: self.ndim,
        }
    }
}

impl std::ops::Div<Tensor> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: Tensor) -> Self::Output {
        assert!(self.ndim == rhs.ndim, "Tensors DIMS do not match");
        for (idx, (dim_t1, dim_t2)) in self.shape.iter().zip(rhs.shape.iter()).enumerate() {
            assert!(
                dim_t1 == dim_t2,
                "Dimensions do not match at dim number {:?}",
                idx
            );
        }
        Tensor {
            data: self
                .data
                .into_iter()
                .zip(rhs.data.into_iter())
                .map(|(x, y)| x / y)
                .collect(),
            shape: self.shape,
            ndim: self.ndim,
        }
    }
}

impl std::ops::Add<Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Self::Output {
        assert!(self.ndim == rhs.ndim, "Tensors DIMS do not match");
        for (idx, (dim_t1, dim_t2)) in self.shape.iter().zip(rhs.shape.iter()).enumerate() {
            assert!(
                dim_t1 == dim_t2,
                "Dimensions do not match at dim number {:?}",
                idx
            );
        }
        Tensor {
            data: self
                .data
                .into_iter()
                .zip(rhs.data.into_iter())
                .map(|(x, y)| x + y)
                .collect(),
            shape: self.shape,
            ndim: self.ndim,
        }
    }
}

impl std::ops::Sub<Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Self::Output {
        assert!(self.ndim == rhs.ndim, "Tensors DIMS do not match");
        for (idx, (dim_t1, dim_t2)) in self.shape.iter().zip(rhs.shape.iter()).enumerate() {
            assert!(
                dim_t1 == dim_t2,
                "Dimensions do not match at dim number {:?}",
                idx
            );
        }
        Tensor {
            data: self
                .data
                .into_iter()
                .zip(rhs.data.into_iter())
                .map(|(x, y)| x - y)
                .collect(),
            shape: self.shape,
            ndim: self.ndim,
        }
    }
}

impl std::ops::Sub<f64> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f64) -> Self::Output {
        Tensor {
            data: self.data.into_iter().map(|x| x - rhs).collect(),
            shape: self.shape,
            ndim: self.ndim,
        }
    }
}

pub struct TensorIterator {
    data: Vec<f64>,
    shape: Vec<usize>,
    idx: usize,
}

impl TensorIterator {
    pub fn into_tensor(self) -> Tensor {
        let ndim = self.shape.len();
        Tensor {
            data: self.data,
            shape: self.shape,
            ndim,
        }
    }
}

impl Iterator for TensorIterator {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.data.len() {
            self.idx += 1;
            Some(self.data[self.idx - 1])
        } else {
            None
        }
    }
}

impl IntoIterator for Tensor {
    type IntoIter = TensorIterator;
    type Item = f64;

    fn into_iter(self) -> Self::IntoIter {
        TensorIterator {
            data: self.data,
            shape: self.shape,
            idx: 0,
        }
    }
}

impl Default for Tensor {
    fn default() -> Self {
        Self::new()
    }
}

// fn copy_data_recursive(
//     &self,
//     dest: &mut Tensor,
//     index: &mut Vec<usize>,
//     dim: usize,
// ) -> Result<(), TensorError> {
//     if dim == self.ndim {
//         if let Some(value) = self.get_element(index)? {
//             dest.set_element(index, *value);
//         }
//     } else {
//         for i in 0..self.shape[dim] {
//             index[dim] = i;
//             self.copy_data_recursive(dest, index, dim + 1);
//         }
//     }
//     Ok(())
// }

// pub fn stack(tensors: &[Self], mut axis: isize) -> Option<Self> {
//     if tensors.is_empty() {
//         return None;
//     }

//     // Check if all tensors have the same shape
//     let first_shape = &tensors[0].shape;
//     for tensor in &tensors[1..] {
//         if &tensor.shape != first_shape {
//             return None;
//         }
//     }

//     // Handle negative axis values
//     let ndim = first_shape.len() as isize;
//     if axis < 0 {
//         axis += ndim + 1;
//     }
//     if axis < 0 || axis as usize > ndim as usize {
//         return None;
//     }

//     // Create a new shape with an additional dimension
//     let mut new_shape = first_shape.clone();
//     new_shape.insert(axis as usize, tensors.len());

//     // Create a new tensor with the new shape and fill it with zeros
//     let mut result = Tensor::zeros(&new_shape);

//     // Copy data from each input tensor to the new tensor
//     for (i, tensor) in tensors.iter().enumerate() {
//         let mut index = vec![0; new_shape.len()];
//         index[axis as usize] = i;
//         Self::copy_data_recursive(tensor, &mut result, &mut index, 0);
//     }

//     Some(result)
// }
