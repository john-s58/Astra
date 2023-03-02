use crate::astra_net::activation::Activation;

use ndarray_rand::rand_distr::{Distribution, Normal};

use crate::tensor::Tensor;

pub trait Layer {
    fn feed_forward(&mut self, inputs: &Tensor) -> Tensor;

    fn back_propagation(&mut self, error: Tensor, learning_rate: f64) -> Tensor;
}

pub struct LayerDense {
    size: usize,
    weights: Tensor,
    biases: Tensor,
    activation: Box<dyn Activation>,
    input: Option<Tensor>,
    output: Option<Tensor>,
}
impl LayerDense {
    pub fn new(size: usize, input_size: usize, activation: Box<dyn Activation>) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(-1.0, 1.0).unwrap();

        Self {
            size,
            weights: Tensor::from_vec(
                (0..size * input_size)
                    .map(|_| normal.sample(&mut rng))
                    .collect(),
                vec![input_size, size],
            ) * (2.0 / (input_size + size) as f64).sqrt(),
            biases: Tensor::from_element(0.0, vec![size]),
            activation,
            input: None,
            output: None,
        }
    }
}

impl Layer for LayerDense {
    fn feed_forward(&mut self, inputs: &Tensor) -> Tensor {
        self.input = Some(inputs.to_owned());

        assert_eq!(
            inputs.len(),
            self.weights.shape[0],
            "input does not match layer input size"
        );

        let inputs_mat = inputs.to_owned().reshape(vec![1, inputs.len()]).unwrap();

        self.output = Some(
            inputs_mat.dot(&self.weights).unwrap()
                + self
                    .biases
                    .clone()
                    .reshape(vec![1, self.biases.len()])
                    .unwrap(),
        );

        self.output = Some(self.activation.call(self.output.clone().unwrap()));

        self.output.clone().unwrap()
    }

    fn back_propagation(&mut self, error: Tensor, learning_rate: f64) -> Tensor {
        let delta_output = self.activation.derive(self.output.clone().unwrap());

        let err = error.clone().reshape(vec![1, error.len()]).unwrap() * delta_output;

        let input_mat = self
            .input
            .clone()
            .unwrap()
            .reshape(vec![1, self.input.clone().unwrap().len()])
            .unwrap();

        let err_mat = err.clone().reshape(vec![1, err.len()]).unwrap();

        let delta_weights = input_mat.transpose().dot(&err_mat).unwrap();
        let delta_biases = err.sum() / err.len() as f64;

        self.weights = self.weights.clone() - (delta_weights * learning_rate);
        self.biases = self.biases.clone() - (delta_biases * learning_rate);

        err.dot(&self.weights.transpose()).unwrap()
    }
}
