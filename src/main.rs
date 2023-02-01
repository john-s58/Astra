use nalgebra::DMatrix;
use rand::Rng;
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};

pub trait Layer {
    fn feed_forward(&mut self, inputs: &Vec<f32>) -> Vec<f32>;

    fn back_propagation(&mut self, error: DMatrix<f32>, learning_rate: f32) -> DMatrix<f32>; 

    fn get_weights(&self) -> DMatrix<f32>;

    fn get_biases(&self) -> DMatrix<f32>;
}

#[derive(Clone, Debug)]
pub struct LayerDense {
    size: usize,
    weights: DMatrix<f32>,
    biases: DMatrix<f32>,
    activation: fn(f32) -> f32,
    activation_derivative: fn(f32) -> f32,
    input: Option<DMatrix<f32>>,
    output: Option<DMatrix<f32>>,
}
impl LayerDense {
    fn new(size: usize, input_size: usize, activation: fn(f32) -> f32, activation_derivative: fn(f32) -> f32) -> Self{
        let mut rng = rand::thread_rng();
        Self {
            size,
            weights: DMatrix::<f32>::from_distribution(input_size, 
            size, &StandardNormal, &mut rng).map(|x| if x == 0.0 {x + 0.3} else {x}),
            biases:  DMatrix::<f32>::from_vec(1, size, vec![0.0f32; size]),
            activation,
            activation_derivative,
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

        self.output = Some((self.input.as_ref().unwrap() *
        &self.weights 
        + self.biases.clone()).map(self.activation));

        self.output.clone().unwrap().data.into()

    }

    fn back_propagation(&mut self, error: DMatrix<f32>, learning_rate: f32) -> DMatrix<f32>{
        let err = error.component_mul(&self.output.as_ref().unwrap().map(self.activation_derivative));

        // println!("self.weights = {:#?}", self.weights.data.as_vec());
        // println!("self.biases = {:#?}", self.biases.data.as_vec());
        // println!("self.output = {:#?}", self.output.clone().unwrap().data.as_vec());
        // println!("err = {:#?}", err.data.as_vec());

        let delta_weights = self.input.as_ref().unwrap().transpose() * &err;
        let delta_biases = &err.row_sum() / err.nrows() as f32;

        self.weights -= delta_weights * learning_rate;
        self.biases -= delta_biases * learning_rate;


        // println!("self.weights POST GRADIENT = {:#?}", self.weights.data.as_vec());
        // println!("self.biases POST GRADIENT = {:#?}", self.biases.data.as_vec());

        err * self.weights.transpose()
    }

    fn get_weights(&self) -> DMatrix<f32> {
        self.weights.clone()
    }

    fn get_biases(&self) -> DMatrix<f32>{
        self.biases.clone()
    }

}
pub struct Net{
    layers: Vec<Box<dyn Layer>>,
}

impl Net{
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>){
        self.layers.push(layer);
    }

    pub fn feed_forward(&mut self, input: &Vec<f32>) -> Vec<f32> {
        let mut output = input.to_vec();
        for l in self.layers.iter_mut(){
            output = l.feed_forward(&output);
        }
        output
    }

    pub fn back_propagation(&mut self, input: &Vec<f32>, target: &Vec<f32>) {
        let mut output = self.feed_forward(input);
        let mut error: Vec<f32> = target.iter().zip(output.iter()).map(|(y, x)| x - y).collect();

        let mut error_mat = DMatrix::<f32>::from_vec(1, error.len(), error.clone());

        for l in self.layers.iter_mut().rev() {
            error_mat = l.back_propagation(error_mat, 0.01);
        }
    }
}

fn main () {
    let mut l1 = Box::new(LayerDense::new(1, 1, |x| x.max(0.0),
                                                                 |x| if x > 0.0 {1.0} else {0.0}));
    // let mut l2 = Box::new(LayerDense::new(1, 3, |x| x.max(0.0),
    //                                                                 |x| if x > 0.0 {1.0} else {0.0}));


    let mut my_net = Net::new();

    my_net.add_layer(l1);
   //my_net.add_layer(l2);

    let mut inputs: Vec<Vec<f32>> = Vec::new();
    let mut targets: Vec<Vec<f32>> = Vec::new();

    for i in 1..10 {
        inputs.push(vec![i as f32]);
        targets.push(vec![(2 * i) as f32]);
    }

   let test_before = my_net.feed_forward(&inputs[0]);
   println!("{:#?}", test_before);

   for (ins, targ) in inputs.iter().zip(targets.iter()){
        my_net.back_propagation(ins, targ);
   }

   let test_after = my_net.feed_forward(&inputs[0]);
   println!("{:#?}", test_after);

}