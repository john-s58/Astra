use crate::astra_net::activation::Activation;


use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Normal, Distribution};

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
        let normal = Normal::new(-1.0, 1.0).unwrap();
        Self {
            size,
            weights: Array2::from_shape_fn((input_size, size), |(_, _)| normal.sample(&mut rng)),
            //weights: Array2::<f64>::from_shape_fn((input_size, size), |(_, _)| rng.gen_range(-1.0..1.0)),
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

        if self.output.as_ref().unwrap().to_vec().iter().any(|v| v.is_nan()){
            println!("WEIGHTS {:#?}", self.weights);
            println!("BIASES {:#?}", self.biases);
            println!("INPUTS {:#?}", inputs);
            panic!("Feed Forward Failed");
        }

        println!("WEIGHTS {:#?}", self.weights);
        println!("INPUTS {:#?}", inputs);
        println!("FEED FORWARD LAYER OUTPUT PRE ACTIVATION {:#?} \n\n", self.output.as_ref().unwrap());

        self.output =Some(self.activation.call(self.output.clone().unwrap()));

        if self.output.as_ref().unwrap().to_vec().iter().any(|v| v.is_nan()){
            println!("WEIGHTS {:#?}", self.weights);
            println!("BIASES {:#?}", self.biases);
            println!("INPUTS {:#?}", inputs);
            self.activation.print_self();
            panic!("Activation Function caused nan");
        }

        self.output.clone().unwrap()

    }

    fn back_propagation(&mut self, error: Array1<f64>, learning_rate: f64) -> Array1<f64>{
        println!("error {:#?} \n\n",error);


        let delta_output = self.activation.derive(self.output.clone().unwrap());
        let err = error.clone() * delta_output;

        let input_mat = Array2::from_shape_vec((1, self.input.as_ref().unwrap().len()), self.input.as_ref().unwrap().to_vec()).unwrap();
        let err_mat = Array2::from_shape_vec((1, err.len()), err.to_vec()).unwrap();

        let delta_weights = input_mat.t().dot(&err_mat);
        let delta_biases = &err.sum() / err.len() as f64;

        

        println!("err {:#?} \n\n",err);

        println!("err_mat {:#?} \n\n",err_mat);


        println!("WEIGHTS PRE CHANGE {:#?} \n\n", self.weights);
        println!("BIASES PRE CHANGE {:#?} \n\n", self.biases);


        println!("DELTA WEIGHTS {:#?} \n\n", delta_weights);
        println!("DELTA BIASES {:#?} \n\n", delta_biases);

        self.weights = self.weights.clone() - (delta_weights * learning_rate);
        self.biases = self.biases.clone() - (delta_biases * learning_rate);

        println!("WEIGHTS POST CHANGE {:#?} \n\n", self.weights);
        println!("BIASES POST CHANGE {:#?} \n\n", self.biases);

        if self.weights.iter().any(|v| v.is_nan()) {
            panic!("THE FUCK IS NAN");
        }


        let err_weights_t = err.dot(&self.weights.t());

        if err_weights_t.iter().any(|v| v.is_nan()) {
            panic!("THE FUCK IS NAN");
        }
        println!("BACKPROP return {:#?}", err_weights_t);

        err_weights_t
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



