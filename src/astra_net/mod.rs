pub mod layer;
pub mod activation;
use crate::astra_net::layer::Layer;


use nalgebra::DMatrix;



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
