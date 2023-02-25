use ndarray::Array1;

pub trait Activation {
    fn call(&self, x: Array1<f64>) -> Array1<f64>;

    fn derive(&self, x: Array1<f64>) -> Array1<f64>;

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
    fn call(&self, x: Array1<f64>) -> Array1<f64> {
        x.mapv(|n| if n > 0.0 { n } else { self.alpha * n })
    }

    fn derive(&self, x: Array1<f64>) -> Array1<f64> {
        x.mapv(|n| if n > 0.0 { 1.0 } else { self.alpha })
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
    fn call(&self, x: Array1<f64>) -> Array1<f64> {
        let input_exp = x.mapv(|n| n.exp());
        input_exp.clone() / input_exp.sum()
    }

    fn derive(&self, x: Array1<f64>) -> Array1<f64> {
        let sm = self.call(x);
        sm.mapv(|n| n * (1.0 - n))
    }

    fn print_self(&self) {
        println!("Softmax");
    }
}
