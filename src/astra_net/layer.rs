use crate::astra_net::activation::Activation;

use rand::Rng;

use ndarray::{Array, Array1, Array2, Array3, ArrayView, ShapeBuilder, array, Zip};

pub trait Layer {
    fn feed_forward(&mut self, inputs: &Array1<f64>) -> Array1<f64>;

    fn back_propagation(&mut self, error: Array1<f64>, learning_rate: f64) -> Array1<f64>; 
}

pub struct LayerDense {
    size: usize,
    weights: Array2<f64>,
    biases: Array1<f64>,
    activation: Box<dyn Activation>,
    input: Option<Array1<f64>>,
    output: Option<Array1<f64>>,
}
impl LayerDense {
    pub fn new(size: usize, input_size: usize, activation: Box<dyn Activation>) -> Self{
        let mut rng = rand::thread_rng();
        Self {
            size,
            weights: Array2::<f64>::from_shape_fn((input_size, size), |(_, _)| rng.gen_range(-1.0..1.0)),
            biases:  Array1::<f64>::from_elem(size, 0.0f64),
            activation,
            input: None,
            output: None,
        }
    }
}

impl Layer for LayerDense{
    fn feed_forward(&mut self, inputs: &Array1<f64>) -> Array1<f64>{

        self.input = Some(inputs.to_owned());

        assert_eq!(inputs.len(), self.weights.nrows(), "input does not match layer input size");

        self.output = Some(inputs.dot(&self.weights)
        + self.biases.clone());

        self.output =Some(self.activation.call(self.output.clone().unwrap()));

        self.output.clone().unwrap()

    }

    fn back_propagation(&mut self, error: Array1<f64>, learning_rate: f64) -> Array1<f64>{


        let delta_output = self.activation.derive(self.output.clone().unwrap());
        let err = error * delta_output;

        let input_mat = Array2::from_shape_vec((1, self.input.as_ref().unwrap().len()), self.input.as_ref().unwrap().to_vec()).unwrap();
        let err_mat = Array2::from_shape_vec((1, err.len()), err.to_vec()).unwrap();

        let delta_weights = input_mat.t().dot(&err_mat);
        let delta_biases = &err.sum() / err.len() as f64;


        self.weights = self.weights.clone() - (delta_weights * learning_rate);
        self.biases = self.biases.clone() - (delta_biases * learning_rate);


        err.dot(&self.weights.t())
    }

}


// pub struct Conv2D {
//     input_shape:  (usize, usize, usize),
//     num_filters: usize,
//     filter_shape: (usize, usize),
//     activation: Option<Box<dyn Activation>>,
//     filters: DMatrix<f64>,
//     bias: DMatrix<f64>,
//     feature_map: Option<DMatrix<f64>>,
// }

// impl Conv2D {
//     pub fn new(input_shape: (usize, usize, usize),num_filters: usize, filter_shape: (usize, usize), activation: Option<Box<dyn Activation>>,) -> Self {
//         Self {
//             input_shape,
//             num_filters,
//             filter_shape,
//             activation,
//             filters: DMatrix::from_fn(num_filters, filter_shape.0 * filter_shape.1, |_, _| rand::random::<f64>()),
//             bias: DMatrix::from_element(1, num_filters, 0.0f64),
//             feature_map: None,
//         }
//     }
// }



