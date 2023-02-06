

#[derive(Clone, Debug)]
pub struct Tensor {
    data: Vec<f64>,
    shape: Vec<u64>,
    ndim: usize,
}

// row first [3, 4] -> 3 rows 4 columns
// [2, 3, 4] -> 2 instances of 3 rows and 4 columns
impl Tensor {
    pub fn new() -> Self {
        Self{
            data: Vec::new(),
            shape: Vec::new(),
            ndim: 0,
        }
    }

    pub fn from_vec(data: Vec<f64>, shape: Vec<u64>) -> Self {
        Self{
            data,
            shape: shape.clone(),
            ndim: shape.len(),
        }
    }

    pub fn transpose(&self) -> Self {
        match self.ndim {
            0 => {panic!("Empty Tenosr Tranpose");},
            1 => {self.to_owned()},
            2 => {
                self.to_owned()
            },
            _ => {
                self.to_owned()
            },
        }
    }

    pub fn get_element(&self, position: Vec<u64>) -> f64 {
        assert_eq!(position.len(), self.ndim, "Position Number Of Dimensions Does Not Match Tensors ndim");
        for (dim_size, pos) in self.shape.iter().zip(position.iter()){
            assert!(dim_size >  pos, "condition dim_size {} > pos {} is false", dim_size, pos);
        }
        let mut position_in_vec: u64 = 0;

        for (i, v) in position.iter().enumerate() {
            if i < self.ndim - 1 {
                position_in_vec += v * self.shape.clone().into_iter().skip(i+1).reduce(|x, y| x * y).unwrap();
            }
            else {
                position_in_vec += v
            }
        }
        assert!(position_in_vec < self.data.len() as u64, "Out Of Bound");

        self.data[position_in_vec as usize].to_owned()
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
            return self.data[0]
        }
        self.data.clone().into_iter().reduce(|x, y| x + y).unwrap()
    }
}