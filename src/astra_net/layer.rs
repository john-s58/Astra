use crate::astra_net::activation::Activation;

use nalgebra::DMatrix;
use rand::Rng;
use rand_distr::StandardNormal;

pub trait Layer {
    fn feed_forward(&mut self, inputs: &Vec<f32>) -> Vec<f32>;

    fn back_propagation(&mut self, error: DMatrix<f32>, learning_rate: f32) -> DMatrix<f32>; 

    fn get_weights(&self) -> DMatrix<f32>;

    fn get_biases(&self) -> DMatrix<f32>;
}

pub struct LayerDense {
    size: usize,
    weights: DMatrix<f32>,
    biases: DMatrix<f32>,
    activation: Box<dyn Activation>,
    input: Option<DMatrix<f32>>,
    output: Option<DMatrix<f32>>,
}
impl LayerDense {
    pub fn new(size: usize, input_size: usize, activation: Box<dyn Activation>) -> Self{
        let mut rng = rand::thread_rng();
        Self {
            size,
            weights: DMatrix::<f32>::from_distribution(input_size, 
            size, &StandardNormal, &mut rng).map(|x| if x == 0.0 {x + 0.3} else {x}),
            biases:  DMatrix::<f32>::from_vec(1, size, vec![0.0f32; size]),
            activation,
            input: None,
            output: None,
        }
    }
    fn into_box(self) -> Box<Self> {
        Box::new(self)
    }
}

impl Layer for LayerDense{
    fn feed_forward(&mut self, inputs: &Vec<f32>) -> Vec<f32>{
        self.input = Some(DMatrix::from_vec(1, inputs.len(), inputs.to_vec()));
        assert_eq!(inputs.len(), self.weights.nrows(), "input does not match layer input size");

        self.output = Some((self.input.as_ref().unwrap() *
        &self.weights 
        + self.biases.clone()));

        self.output =Some(self.activation.call(self.output.clone().unwrap()));

        self.output.clone().unwrap().data.into()

    }

    fn back_propagation(&mut self, error: DMatrix<f32>, learning_rate: f32) -> DMatrix<f32>{


        let delta_output = self.activation.derive(self.output.clone().unwrap());
        let err = error.component_mul(&delta_output);


        let delta_weights = self.input.as_ref().unwrap().transpose() * &err;
        let delta_biases = &err.row_sum() / err.nrows() as f32;

        self.weights -= delta_weights * learning_rate;
        self.biases -= delta_biases * learning_rate;


        err * self.weights.transpose()
    }

    fn get_weights(&self) -> DMatrix<f32> {
        self.weights.clone()
    }

    fn get_biases(&self) -> DMatrix<f32>{
        self.biases.clone()
    }

}