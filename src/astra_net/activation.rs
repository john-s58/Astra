use ndarray::{Array, Array1, Array2, Array3, ArrayView, ShapeBuilder, array};

pub trait Activation {
    fn call(&self, x: Array1<f64>) -> Array1<f64>;

    fn derive(&self, x: Array1<f64>) -> Array1<f64>;
}

pub struct LeakyReLU;

impl LeakyReLU {
    pub fn new() -> Self{
        Self
    }
}

impl Activation for LeakyReLU {
    fn call(&self, x: Array1<f64>) -> Array1<f64>{
        x.mapv(|n| if n > 0.0 {n} else {0.33 * n})
    }

    fn derive(&self, x: Array1<f64>) -> Array1<f64>{
        x.mapv(|n| if n > 0.0 {1.0} else {0.33})

    }

}

pub struct Softmax;

impl Softmax {
    pub fn new() -> Self{
        Self
    }
}

impl Activation for Softmax {
    fn call(&self, x: Array1<f64>) -> Array1<f64>{
        let input_exp = x.mapv(|n| n.exp());
        input_exp.clone() / input_exp.sum()
    }

    fn derive(&self, x: Array1<f64>) -> Array1<f64>{
        let sm = self.call(x);
        sm.mapv(|n| n * (1.0 - n))
    }

}