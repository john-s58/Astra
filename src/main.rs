use crate::astra_net::activation::{LeakyReLU, Softmax};
use crate::astra_net::layer::LayerDense;
use crate::astra_net::Net;

use crate::tensor::Tensor;

mod astra_net;
mod tensor;
use rand::Rng;


fn main() {
    let l1 = Box::new(LayerDense::new(6, 2, Box::new(LeakyReLU::new(0.1))));
    let l2 = Box::new(LayerDense::new(6, 6, Box::new(LeakyReLU::new(0.1))));
    let l3 = Box::new(LayerDense::new(12, 6, Box::new(LeakyReLU::new(0.1))));
    let l4 = Box::new(LayerDense::new(6, 12, Box::new(LeakyReLU::new(0.1))));
    let lend = Box::new(LayerDense::new(2, 6, Box::new(Softmax::new())));

    let mut my_net = Net::new();
    my_net.set_learning_rate(0.001);

    my_net.add_layer(l1);
    my_net.add_layer(l2);
    my_net.add_layer(l3);
    my_net.add_layer(l4);
    my_net.add_layer(lend);

    let input_data = generate_2d_cluster_dataset(5000);

    let target_data: Vec<Tensor> = (0..input_data.len())
    .map(|x| if x < input_data.len()/2 {Tensor::from_vec(vec![1.0, 0.0], vec![2])} 
                    else {Tensor::from_vec(vec![0.0, 1.0], vec![2])}).collect();

    for (input, target) in input_data.into_iter().zip(target_data.into_iter()) {
        my_net.back_propagation(&input, &target);
    }

    let test1 = my_net.feed_forward(&Tensor::from_vec(vec![-3.0, -3.0], vec![2]));

    println!("test1, should be 1 , 0  {:#?}", test1);

    let test2 = my_net.feed_forward(&Tensor::from_vec(vec![8.0, 8.0], vec![2]));

    println!("test2, should be 0 , 1  {:#?}", test2);
   
}


fn generate_2d_cluster_dataset(num_samples: usize) -> Vec<Tensor> {
    let mut rng = rand::thread_rng();
    let mut data: Vec<Tensor> = Vec::with_capacity(num_samples);
    for _ in 0..num_samples / 2 {
        let x1 = rng.gen_range(-5.0..5.0);
        let x2 = rng.gen_range(-5.0..5.0);
        data.push(Tensor::from_vec(vec![x1, x2], vec![2]));
    }
    for _ in num_samples / 2..num_samples {
        let x1 = rng.gen_range(5.0..10.0);
        let x2 = rng.gen_range(5.0..10.0);
        data.push(Tensor::from_vec(vec![x1, x2], vec![2]));
    }
    data
}

// fn generate_3d_cluster_dataset(num_samples: usize) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
//     let mut rng = rand::thread_rng();
//     let mut x_data: Vec<Array1<f64>> = Vec::with_capacity(num_samples);
//     let mut y_data: Vec<Array1<f64>> = Vec::with_capacity(num_samples);

//     for i in 0..num_samples {
//         if i % 3 == 1 {
//             let x1 = rng.gen_range(-1.0..0.0);
//             let x2 = rng.gen_range(-1.0..0.0);
//             x_data.push(array![x1, x2]);
//             y_data.push(array![1.0, 0.0, 0.0]);
//         }
//         if i % 3 == 2 {
//             let x1 = rng.gen_range(0.0..1.0);
//             let x2 = rng.gen_range(0.0..1.0);
//             x_data.push(array![x1, x2]);
//             y_data.push(array![0.0, 1.0, 0.0]);
//         }
//         if i % 3 == 0 {
//             let x1 = rng.gen_range(1.0..2.0);
//             let x2 = rng.gen_range(1.0..2.0);
//             x_data.push(array![x1, x2]);
//             y_data.push(array![0.0, 0.0, 1.0]);
//         }
//     }
//     (x_data, y_data)
// }
