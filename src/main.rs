use std::f32::consts::PI;
use core::fmt::Debug;
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

pub trait Activation {
    fn call(&self, x: DMatrix<f32>) -> DMatrix<f32>;

    fn derive(&self, x: DMatrix<f32>) -> DMatrix<f32>;
}

struct Leaky_ReLU;

impl Leaky_ReLU {
    fn new() -> Self{
        Self
    }
    fn into_box(self) -> Box<Self> {
        Box::new(self)
    }
}

impl Activation for Leaky_ReLU {
    fn call(&self, x: DMatrix<f32>) -> DMatrix<f32>{
        x.map(|n| if n > 0.0 {n} else {0.33 * n})
    }

    fn derive(&self, x: DMatrix<f32>) -> DMatrix<f32>{
        x.map(|n| if n > 0.0 {1.0} else {0.33})

    }

}

struct Softmax;

impl Softmax {
    fn new() -> Self{
        Self
    }
    fn into_box(self) -> Box<Self> {
        Box::new(self)
    }
}

impl Activation for Softmax {
    fn call(&self, x: DMatrix<f32>) -> DMatrix<f32>{
        let input_exp = x.exp();
        input_exp.clone() / input_exp.sum()
    }

    fn derive(&self, x: DMatrix<f32>) -> DMatrix<f32>{
        let sm = self.call(x);
        sm.map(|n| n * (1.0 - n))
    }

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
    fn new(size: usize, input_size: usize, activation: Box<dyn Activation>) -> Self{
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

        for l in self.layers.iter_mut().rev() {;
            error_mat = l.back_propagation(error_mat, 0.01);
        }
    }
}

fn main () {
    let mut l1 = Box::new(LayerDense::new(6, 2, Box::new(Leaky_ReLU::new())));
    let mut l2 = Box::new(LayerDense::new(18, 6, Box::new(Leaky_ReLU::new())));
    let mut l3 = Box::new(LayerDense::new(3, 18, Box::new(Softmax::new())));

    let mut my_net = Net::new();

    my_net.add_layer(l1);
    my_net.add_layer(l2);
    my_net.add_layer(l3);

    let mut inputs: Vec<Vec<f32>> = Vec::new();
    let mut targets: Vec<Vec<f32>> = Vec::new();

    // for _ in 0..100 {
    //     for i in 1..10 {
    //         inputs.push(vec![i as f32]);
    //         targets.push(vec![(2 * i) as f32]);
    //     }
    // }

    // inputs = generate_2d_cluster_dataset(100000);
    // for i in  0..inputs.len(){
    //     targets.push(vec![if i < inputs.len() / 2 {1.0} else {0.0}, if i < inputs.len() / 2 {0.0} else {1.0}]);
    // }

    inputs = generate_3d_cluster_dataset(100000);
    for _ in 0..inputs.len() / 3 {
        targets.push(vec![1.0, 0.0, 0.0]);
    }
    for _ in inputs.len() / 3..2*inputs.len() / 3 {
        targets.push(vec![0.0, 1.0, 0.0]);
    }
    for _ in 2*inputs.len() / 3..inputs.len() {
        targets.push(vec![0.0, 0.0, 1.0]);
    }

    let test_before = my_net.feed_forward(&inputs[0]);
    println!("{:#?}", test_before);

    let test_before2 = my_net.feed_forward(&inputs[50000]);
    println!("{:#?}", test_before2);

    let test_before3 = my_net.feed_forward(&inputs[99000]);
    println!("{:#?}", test_before3);

    for (ins, targ) in inputs.iter().zip(targets.iter()){
            my_net.back_propagation(ins, targ);
    }

    let test_after = my_net.feed_forward(&inputs[0]);
    println!("{:#?}", test_after);

    println!("inputs 50k {:#?}", &inputs[50000]);
    println!("targets 50k {:#?}", &targets[50000]);
    let test_after2 = my_net.feed_forward(&inputs[50000]);
    println!("predict after 50k {:#?}", test_after2);

    let test_after3 = my_net.feed_forward(&inputs[99000]);
    println!("{:#?}", test_after3);

}



fn generate_2d_cluster_dataset(num_samples: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(num_samples);
    for _ in 0..num_samples / 2 {
        let x1 = rng.gen_range(-5.0..5.0);
        let x2 = rng.gen_range(-5.0..5.0);
        data.push(vec![x1, x2]);
    }
    for _ in num_samples / 2..num_samples {
        let x1 = rng.gen_range(5.0..15.0);
        let x2 = rng.gen_range(5.0..15.0);
        data.push(vec![x1, x2]);
    }
    data
}


fn generate_3d_cluster_dataset(num_samples: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(num_samples);
    for _ in 0..num_samples / 3 {
        let x1 = rng.gen_range(1.0..5.0);
        let x2 = rng.gen_range(1.0..5.0);
        data.push(vec![x1, x2]);
    }
    for _ in num_samples / 3..2*num_samples / 3 {
        let x1 = rng.gen_range(6.0..10.0);
        let x2 = rng.gen_range(6.0..10.0);
        data.push(vec![x1, x2]);
    }
    for _ in 2*num_samples / 3..num_samples {
        let x1 = rng.gen_range(11.0..15.0);
        let x2 = rng.gen_range(11.0..15.0);
        data.push(vec![x1, x2]);
    }
    data
}