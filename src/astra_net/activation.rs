use crate::tensor::Tensor;

pub trait Activation {
    fn call(&self, x: Tensor) -> Tensor;

    fn derive(&self, x: Tensor) -> Tensor;

    fn print_self(&self);
}

#[derive(Clone)]
pub struct LeakyReLU {
    alpha: f64,
}

impl LeakyReLU {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Activation for LeakyReLU {
    fn call(&self, x: Tensor) -> Tensor {
        x.map(|n| if n > 0.0 { n } else { self.alpha * n })
    }

    fn derive(&self, x: Tensor) -> Tensor {
        x.map(|n| if n > 0.0 { 1.0 } else { self.alpha })
    }
    fn print_self(&self) {
        println!("LeakyReLU");
    }
}

#[derive(Clone)]
pub struct Softmax;

impl Softmax {
    pub fn new() -> Self {
        Self
    }
}

impl Activation for Softmax {
    fn call(&self, x: Tensor) -> Tensor {
        let input_exp = x.map(|n| n.exp());
        input_exp.clone() / input_exp.sum()
    }

    fn derive(&self, x: Tensor) -> Tensor {
        let sm = self.call(x);
        sm.map(|n| n * (1.0 - n))
    }

    fn print_self(&self) {
        println!("Softmax");
    }
}

pub struct Sigmoid;

impl Activation for Sigmoid {
    fn call(&self, x: Tensor) -> Tensor {
        x.map(|n| 1. / (1. + (-n).exp()))
    }
    fn derive(&self, x: Tensor) -> Tensor {
        let s = self.call(x);
        s.clone() * (s.map(|n| 1. - n))
    }
    fn print_self(&self) {
        println!("Sigmoid");
    }
}

pub struct TanH;

impl Activation for TanH {
    fn call(&self, x: Tensor) -> Tensor {
        todo!()
    }
    fn derive(&self, x: Tensor) -> Tensor {
        todo!()
    }
    fn print_self(&self) {
        println!("TanH");
    }
}
