use nalgebra::DMatrix;

pub trait Activation {
    fn call(&self, x: DMatrix<f32>) -> DMatrix<f32>;

    fn derive(&self, x: DMatrix<f32>) -> DMatrix<f32>;
}

pub struct LeakyReLU;

impl LeakyReLU {
    pub fn new() -> Self{
        Self
    }
}

impl Activation for LeakyReLU {
    fn call(&self, x: DMatrix<f32>) -> DMatrix<f32>{
        x.map(|n| if n > 0.0 {n} else {0.33 * n})
    }

    fn derive(&self, x: DMatrix<f32>) -> DMatrix<f32>{
        x.map(|n| if n > 0.0 {1.0} else {0.33})

    }

}

pub struct Softmax;

impl Softmax {
    pub fn new() -> Self{
        Self
    }
}

impl Activation for Softmax {
    fn call(&self, x: DMatrix<f32>) -> DMatrix<f32>{
        let input_exp = x.exp();
        input_exp.clone() / input_exp.sum()
    }

    fn derive(&self, x: DMatrix<f32>) -> DMatrix<f32>{
        let sm = self.call(x);
        sm.map(|n| n * (1.0 - n))
    }

}