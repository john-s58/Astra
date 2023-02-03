use crate::astra_net::layer::LayerDense;
use crate::astra_net::activation::{LeakyReLU, Softmax};
use crate::astra_net::Net;

mod astra_net;

use rand::Rng;
use ndarray::{self, ShapeBuilder};

fn main () {
    let l1 = Box::new(LayerDense::new(6, 2, Box::new(LeakyReLU::new())));
    let l2 = Box::new(LayerDense::new(18, 6, Box::new(LeakyReLU::new())));
    let l3 = Box::new(LayerDense::new(2, 18, Box::new(Softmax::new())));

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

    inputs = generate_2d_cluster_dataset(100);
    for i in  0..inputs.len(){
        targets.push(vec![if i < inputs.len() / 2 {1.0} else {0.0}, if i < inputs.len() / 2 {0.0} else {1.0}]);
    }

    // inputs = generate_3d_cluster_dataset(100000);
    // for _ in 0..inputs.len() / 3 {
    //     targets.push(vec![1.0, 0.0, 0.0]);
    // }
    // for _ in inputs.len() / 3..2*inputs.len() / 3 {
    //     targets.push(vec![0.0, 1.0, 0.0]);
    // }
    // for _ in 2*inputs.len() / 3..inputs.len() {
    //     targets.push(vec![0.0, 0.0, 1.0]);
    // }




    for (ins, targ) in inputs.iter().zip(targets.iter()){
            my_net.back_propagation(ins, targ);
    }

    // let my_test_c1 =  my_net.feed_forward(&vec![1.5f32, -3.7]);
    // println!("should be cluster 1: {:#?}", my_test_c1);

    // let my_test_c2 =  my_net.feed_forward(&vec![10.5f32,13.7]);
    // println!("should be cluster 2: {:#?}", my_test_c2);



    let test = ndarray::Array2::from_elem(ShapeBuilder::into_shape((2, 3)), 1.0);
    println!("{:#?}", test);
    let transposed = test.t();
    println!("{:#?}", transposed);


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