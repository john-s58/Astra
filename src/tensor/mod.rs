#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
    pub ndim: usize,
}

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

    pub fn from_vec(data: Vec<f64>, shape: Vec<usize>) -> Self {
        assert!(!shape.contains(&0), "shape contains 0");
        assert_eq!(
            data.len(),
            shape.iter().product(),
            "shape does not match data"
        );

        Self {
            data,
            shape: shape.clone(),
            ndim: shape.len(),
        }
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

    pub fn from_fn(shape: Vec<usize>, func: impl Fn() -> f64) -> Self {
        Self {
            data: (0..shape.iter().product()).map(|_| func()).collect(),
            shape: shape.clone(),
            ndim: shape.len(),
        }
    }

    pub fn reshape(self, new_shape: Vec<usize>) -> Option<Self> {
        assert!(!new_shape.contains(&0), "shape contains 0");
        if self.shape.clone().into_iter().reduce(|a, b| a * b).unwrap()
            != new_shape.clone().into_iter().reduce(|a, b| a * b).unwrap()
        {
            return None;
        }
        Some(Self {
            data: self.data,
            shape: new_shape.clone(),
            ndim: new_shape.len(),
        })
    }

    pub fn identity(shape: Vec<usize>) -> Option<Self> {
        if shape.windows(2).all(|w| w[0] == w[1]) == false {
            return None;
        }
        let ndim = shape.len();
        let elem_size = shape[0];
        let mut res = Tensor::from_element(0.0, shape);
        for i in 0..elem_size {
            *res.get_element_mut(&vec![i; ndim]).unwrap() = 1.0;
        }
        Some(res)
    }

    pub fn matrix(m: usize, n: usize, data: Vec<f64>) -> Option<Self> {
        match m * n == data.len() {
            false => None,
            true => Some(Self {
                data,
                shape: vec![m, n],
                ndim: 2,
            }),
        }
    }

    pub fn transpose(&self) -> Self {
        match self.ndim {
            0 => {
                panic!("Empty Tenosr Tranpose");
            }
            1 => self.to_owned(),
            2 => {
                let mut transposed = self.to_owned();
                transposed.shape = transposed.shape.into_iter().rev().collect();
                for i in 0..self.shape[1] {
                    for j in 0..self.shape[0] {
                        *transposed.get_element_mut(&[i, j]).unwrap() =
                            self.get_element(&[j, i]).unwrap().to_owned();
                    }
                }
                transposed
            }
            _ => self.to_owned(),
        }
    }

    pub fn get_index(&self, indices: &[usize]) -> usize {
        let mut index = 0;
        let mut stride = 1;
        for i in 0..self.shape.len() {
            index += indices[i] * stride;
            stride *= self.shape[i];
        }
        index
    }

    pub fn get_indices(&self, index: usize) -> Option<Vec<usize>> {
        if index >= self.data.len() {
            return None;
        }

        let mut indices = vec![0; self.shape.len()];
        let mut remainder = index;

        for i in (0..self.shape.len()).rev() {
            let stride = self.shape[i];
            indices[i] = remainder % stride;
            remainder /= stride;
        }

        Some(indices)
    }

    pub fn get_element(&self, indices: &[usize]) -> Option<&f64> {
        let index: usize = self.get_index(indices);
        self.data.get(index)
    }

    pub fn get_element_mut(&mut self, indices: &[usize]) -> Option<&mut f64> {
        let index: usize = self.get_index(indices);
        self.data.get_mut(index)
    }

    pub fn set_element(&mut self, indices: &[usize], value: f64) {}

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
        self.data.clone().into_iter().reduce(|x, y| x + y).unwrap()
    }

    pub fn dot(&self, other: &Self) -> Option<Self> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return None;
        }
        if self.shape[1] != other.shape[0] {
            return None;
        }
        let mut result = Tensor::from_element(0.0, vec![self.shape[0], other.shape[1]]);

        for i in 0..self.shape[0] {
            for j in 0..other.shape[1] {
                let mut val = 0.0;
                for k in 0..self.shape[1] {
                    val += self.get_element(&[i, k]).unwrap() * other.get_element(&[k, j]).unwrap();
                }
                *result.get_element_mut(&[i, j]).unwrap() = val;
            }
        }

        Some(result)
    }

    pub fn inverse(&self) -> Self {
        todo!()
    }

    pub fn diag(&self) -> Vec<f64> {
        todo!()
    }

    pub fn push_value(&mut self, val: f64) {
        todo!()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn to_vec(&self) -> Vec<f64> {
        self.data.to_owned()
    }

    pub fn get_sub_matrix(&self, left_corner: &[usize], shape: &[usize]) -> Option<Self> {
        // Check if the input is valid, i.e., the tensor is 2D, and the sub-matrix is within bounds
        if self.ndim != 2 || left_corner.len() != 2 || shape.len() != 2 {
            return None;
        }

        let (left_corner_i, left_corner_j) = (left_corner[0], left_corner[1]);
        let (shape_i, shape_j) = (shape[0], shape[1]);

        if left_corner_i + shape_i > self.shape[0] || left_corner_j + shape_j > self.shape[1] {
            return None;
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

        Some(sub_matrix)
    }

    fn copy_recursive(
        &self,
        padded: &mut Tensor,
        index: &mut Vec<usize>,
        ranges: &[Vec<usize>],
        dim: usize,
    ) {
        if dim < self.ndim - 1 {
            for i in &ranges[dim] {
                index[dim] = *i;
                self.copy_recursive(padded, index, ranges, dim + 1);
            }
        } else {
            for i in &ranges[dim] {
                index[dim] = *i;
                let value = *self.get_element(&index[dim - self.ndim + 1..]).unwrap();
                *padded.get_element_mut(&index[..]).unwrap() = value;
            }
        }
    }

    pub fn pad(&self, padding: &[(usize, usize)]) -> Option<Self> {
        // Check if the padding specification has the same length as the number of dimensions
        if padding.len() != self.ndim {
            return None;
        }

        // Calculate the new shape and initialize the padded tensor
        let new_shape: Vec<usize> = self
            .shape
            .iter()
            .zip(padding.iter())
            .map(|(dim_size, (pad_before, pad_after))| dim_size + pad_before + pad_after)
            .collect();

        let mut padded = Tensor::from_element(0.0, new_shape.clone());

        // Copy the data from the original tensor to the padded tensor
        let index_ranges: Vec<Vec<usize>> = self
            .shape
            .iter()
            .zip(padding.iter())
            .map(|(dim_size, (pad_before, _pad_after))| {
                (0..*dim_size).map(|i| i + pad_before).collect()
            })
            .collect();

        let mut index: Vec<usize> = vec![0; self.ndim];
        self.copy_recursive(&mut padded, &mut index, &index_ranges, 0);

        Some(padded)
    }

    fn slice_recursive(
        &self,
        sub_tensor: &mut Tensor,
        index: &mut Vec<usize>,
        ranges: &[(usize, usize)],
        dim: usize,
    ) {
        if dim < self.ndim - 1 {
            for i in ranges[dim].0..ranges[dim].1 {
                index[dim] = i;
                self.slice_recursive(sub_tensor, index, ranges, dim + 1);
            }
        } else {
            for i in ranges[dim].0..ranges[dim].1 {
                index[dim] = i;
                let value = *self.get_element(&index[..]).unwrap();
                *sub_tensor
                    .get_element_mut(
                        &index
                            .iter()
                            .enumerate()
                            .map(|(idx, &val)| val - ranges[idx].0)
                            .collect::<Vec<usize>>()[..],
                    )
                    .unwrap() = value;
            }
        }
    }

    pub fn slice(&self, ranges: &[(usize, usize)]) -> Option<Self> {
        // Check if the ranges specification has the same length as the number of dimensions
        if ranges.len() != self.ndim {
            return None;
        }

        // Check if the ranges are valid and calculate the new shape
        let new_shape: Vec<usize> = self
            .shape
            .iter()
            .zip(ranges.iter())
            .map(|(dim_size, (start, end))| {
                if start >= end || *end > *dim_size {
                    None
                } else {
                    Some(end - start)
                }
            })
            .collect::<Option<Vec<usize>>>()?;

        let mut sub_tensor = Tensor::from_element(0.0, new_shape.clone());

        let mut index: Vec<usize> = vec![0; self.ndim];
        self.slice_recursive(&mut sub_tensor, &mut index, &ranges, 0);

        Some(sub_tensor)
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
