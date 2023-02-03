use crate::astra_net::activation::Activation;

use nalgebra::DMatrix;
use rand_distr::StandardNormal;

pub trait Layer {
    fn feed_forward(&mut self, inputs: &Vec<f32>) -> Vec<f32>;

    fn back_propagation(&mut self, error: DMatrix<f32>, learning_rate: f32) -> DMatrix<f32>; 
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
            biases:  DMatrix::<f32>::from_element(1, size, 0.0f32),
            activation,
            input: None,
            output: None,
        }
    }
}

impl Layer for LayerDense{
    fn feed_forward(&mut self, inputs: &Vec<f32>) -> Vec<f32>{
        self.input = Some(DMatrix::from_vec(1, inputs.len(), inputs.to_vec()));
        assert_eq!(inputs.len(), self.weights.nrows(), "input does not match layer input size");

        self.output = Some(self.input.as_ref().unwrap() *
        &self.weights 
        + self.biases.clone());

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

}


pub struct Conv2D {
    input_shape:  (usize, usize, usize),
    num_filters: usize,
    filter_shape: (usize, usize),
    activation: Option<Box<dyn Activation>>,
    filters: DMatrix<f32>,
    bias: DMatrix<f32>,
    feature_map: Option<DMatrix<f32>>,
}

impl Conv2D {
    pub fn new(input_shape: (usize, usize, usize),num_filters: usize, filter_shape: (usize, usize), activation: Option<Box<dyn Activation>>,) -> Self {
        Self {
            input_shape,
            num_filters,
            filter_shape,
            activation,
            filters: DMatrix::from_fn(num_filters, filter_shape.0 * filter_shape.1, |_, _| rand::random::<f32>()),
            bias: DMatrix::from_element(1, num_filters, 0.0f32),
            feature_map: None,
        }
    }
}



