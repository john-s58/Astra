#[derive(Clone, Debug)]
pub struct Tensor {
    data: Vec<f64>,
    shape: Vec<usize>,
    ndim: usize,
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
        assert_eq!(data.len(),
            shape.clone().into_iter().reduce(|a, b| a*b).unwrap() as usize,
            "shape does not match data");

        Self {
            data,
            shape: shape.clone(),
            ndim: shape.len(),
        }
    }

    pub fn from_element(element: f64, shape: Vec<usize>) -> Self {
        assert!(!shape.contains(&0), "shape contains 0");

        let data_size = shape.clone().into_iter().reduce(|x, y| x * y).unwrap() as usize;
        Tensor {
            data: vec![element; data_size],
            shape: shape.clone(),
            ndim: shape.len(),
        }
    }

    pub fn reshape(self, new_shape: Vec<usize>) -> Option<Tensor>{
        assert!(!new_shape.contains(&0), "shape contains 0");
        if self.shape.clone().into_iter().reduce(|a, b| a*b).unwrap() != new_shape.clone().into_iter().reduce(|a, b| a*b).unwrap() {
            return None;
        }
        Some(Tensor { data: self.data, shape: new_shape.clone(), ndim: new_shape.len() })
    }

    pub fn identity(shape: Vec<usize>) -> Self {
        // let mut res = Tensor::from_element(0.0, shape);
        // res
        todo!()
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
                        *transposed.get_element_mut(&[i, j]).unwrap() = self.get_element(&[j, i]).unwrap().to_owned();
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

    pub fn set_element(&mut self, indices: &[usize], value: f64){

    }

    pub fn map(self, fun: fn(f64) -> f64) -> Self {
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
