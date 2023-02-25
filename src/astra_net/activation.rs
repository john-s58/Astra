use crate::tensor::Tensor;

pub trait Activation {
    fn call(&self, x: Tensor) -> Tensor;

    fn derive(&self, x: Tensor) -> Tensor;

    fn print_self(&self);
}

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
        Tensor::from_vec(x.clone()
                                .to_vec()
                                .into_iter()
                                .map(|n| if n > 0.0 { n } else { self.alpha * n })
                                .collect()
                                , x.shape)
    }

    fn derive(&self, x: Tensor) -> Tensor {
        Tensor::from_vec(x.clone()
                                .to_vec()
                                .into_iter()
                                .map(|n| if n > 0.0 { 1.0 } else { self.alpha })
                                .collect()
                                , x.shape)
    }

    fn print_self(&self) {
        println!("LeakyReLU");
    }
}

pub struct Softmax;

impl Softmax {
    pub fn new() -> Self {
        Self
    }
}

impl Activation for Softmax {
    fn call(&self, x: Tensor) -> Tensor {
        let input_exp = Tensor::from_vec(x.clone()
                                                        .to_vec()
                                                        .into_iter()
                                                        .map(|n| n.exp())
                                                        .collect()
                                                        , x.shape);
        
        input_exp.clone() / input_exp.sum()
    }

    fn derive(&self, x: Tensor) -> Tensor {
        let sm = self.call(x);
        Tensor::from_vec(sm.clone()
                                .to_vec()
                                .into_iter()
                                .map(|n| n * (1.0 - n))
                                .collect()
                                , sm.shape)
    }

    fn print_self(&self) {
        println!("Softmax");
    }
}
