use crate::astra_net::layer::LayerDense;
use crate::astra_net::activation::{LeakyReLU, Softmax};
use crate::astra_net::Net;

mod astra_net;

use rand::Rng;
use ndarray::{Array1, Array2, array};

fn main () {
    let l1 = Box::new(LayerDense::new(6, 2, Box::new(LeakyReLU::new(0.1))));
    let l2 = Box::new(LayerDense::new(6, 6, Box::new(LeakyReLU::new(0.1))));
    let l3 = Box::new(LayerDense::new(3, 6, Box::new(Softmax::new())));

    let mut my_net = Net::new();
    my_net.set_learning_rate(0.001);

    my_net.add_layer(l1);
    my_net.add_layer(l2);
    my_net.add_layer(l3);

    let mut inputs: Vec<Array1<f64>> = Vec::new();
    let mut targets: Vec<Array1<f64>> = Vec::new();


    // inputs = generate_2d_cluster_dataset(30000);
    // for i in  0..inputs.len(){
    //     targets.push(Array1::from(vec![if i < inputs.len() / 2 {1.0} else {0.0}, if i < inputs.len() / 2 {0.0} else {1.0}]));
    // }

    (inputs, targets) = generate_3d_cluster_dataset(160);


    for (ins, targ) in inputs.iter().zip(targets.iter()){
            my_net.back_propagation(ins, targ);
    }
    println!("Cluster 1 test post training Output: {:#?}", my_net.feed_forward(&array![-2f64, 2f64]));
    println!("Cluster 2 test post training Output: {:#?}", my_net.feed_forward(&array![8f64, 7f64]));
    println!("Cluster 3 test post training Output: {:#?}", my_net.feed_forward(&array![13f64, 14f64]));

}



fn generate_2d_cluster_dataset(num_samples: usize) -> Vec<Array1<f64>> {
    let mut rng = rand::thread_rng();
    let mut data: Vec<Array1<f64>> = Vec::with_capacity(num_samples);
    for _ in 0..num_samples / 2 {
        let x1 = rng.gen_range(-5.0..5.0);
        let x2 = rng.gen_range(-5.0..5.0);
        data.push(array![x1, x2]);
    }
    for _ in num_samples / 2..num_samples {
        let x1 = rng.gen_range(5.0..10.0);
        let x2 = rng.gen_range(5.0..10.0);
        data.push(array![x1, x2]);
    }
    data
}

fn generate_3d_cluster_dataset(num_samples: usize) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
    let mut rng = rand::thread_rng();
    let mut x_data: Vec<Array1<f64>> = Vec::with_capacity(num_samples);
    let mut y_data: Vec<Array1<f64>> = Vec::with_capacity(num_samples);

    for i in 0..num_samples{
        if i % 3 == 1 {
            let x1 = rng.gen_range(-5.0..5.0);
            let x2 = rng.gen_range(-5.0..5.0);
            x_data.push(array![x1, x2]);
            y_data.push(array![1.0, 0.0, 0.0]);
        }
        if i % 3 == 2 {
            let x1 = rng.gen_range(5.0..10.0);
            let x2 = rng.gen_range(5.0..10.0);
            x_data.push(array![x1, x2]);
            y_data.push(array![0.0, 1.0, 0.0]);
        }
        if i % 3 == 0 {
            let x1 = rng.gen_range(10.0..15.0);
            let x2 = rng.gen_range(10.0..15.0);
            x_data.push(array![x1, x2]);
            y_data.push(array![0.0, 0.0, 1.0]);
        }

    }
    (x_data, y_data)
}