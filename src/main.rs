// mod astra_net;
// mod error;
// mod tensor;
// mod tensor2;
// mod helper_traits;

// use std::error::Error;

// use crate::astra_net::activation::{LeakyReLU, Sigmoid, Softmax, TanH};
// use crate::astra_net::conv2d::{LayerConv2D, Padding};
// use crate::astra_net::dense::LayerDense;
// use crate::astra_net::flatten::LayerFlatten;
// use crate::astra_net::loss::{CategoricalCrossEntropy, MSE};
// use crate::astra_net::Net;
// use crate::error::AstraError;
// use crate::tensor::Tensor;

// use rand::Rng;

// fn main() {
//     test_image_rec();
// }

// fn test_dense() -> Result<(), Box<dyn Error>> {
//     let l1 = Box::new(LayerDense::new(6, 2, Box::new(LeakyReLU::new(0.1)))?);
//     let l2 = Box::new(LayerDense::new(6, 6, Box::new(LeakyReLU::new(0.1)))?);
//     let l3 = Box::new(LayerDense::new(12, 6, Box::new(LeakyReLU::new(0.1)))?);
//     let l4 = Box::new(LayerDense::new(6, 12, Box::new(LeakyReLU::new(0.1)))?);
//     let lend = Box::new(LayerDense::new(2, 6, Box::new(Softmax::new()))?);

//     let mut my_net = Net::new(Box::new(MSE::new()), 0.001, Some(1.0));
//     my_net.set_learning_rate(0.001);

//     my_net.add_layer(l1);
//     my_net.add_layer(l2);
//     my_net.add_layer(l3);
//     my_net.add_layer(l4);
//     my_net.add_layer(lend);

//     let input_data = generate_2d_cluster_dataset(50000);

//     let target_data: Vec<Tensor> = (0..input_data.len())
//         .map(|x| {
//             if x < input_data.len() / 2 {
//                 Tensor::from_vec(vec![1.0, 0.0], vec![2]).unwrap()
//             } else {
//                 Tensor::from_vec(vec![0.0, 1.0], vec![2]).unwrap()
//             }
//         })
//         .collect();

//     for (input, target) in input_data.into_iter().zip(target_data.into_iter()) {
//         my_net.back_propagation(&input, &target)?;
//     }

//     let test1 = my_net.feed_forward(&Tensor::from_vec(vec![-3.0, -3.0], vec![2])?)?;

//     println!("test1, should be 1 , 0  {:#?}", test1);

//     let test2 = my_net.feed_forward(&Tensor::from_vec(vec![8.0, 8.0], vec![2])?)?;

//     println!("test2, should be 0 , 1  {:#?}", test2);

//     Ok(())
// }

// fn generate_2d_cluster_dataset(num_samples: usize) -> Vec<Tensor> {
//     let mut rng = rand::thread_rng();
//     let mut data: Vec<Tensor> = Vec::with_capacity(num_samples);
//     for _ in 0..num_samples / 2 {
//         let x1 = rng.gen_range(-5.0..5.0);
//         let x2 = rng.gen_range(-5.0..5.0);
//         data.push(Tensor::from_vec(vec![x1, x2], vec![2]).unwrap());
//     }
//     for _ in num_samples / 2..num_samples {
//         let x1 = rng.gen_range(5.0..10.0);
//         let x2 = rng.gen_range(5.0..10.0);
//         data.push(Tensor::from_vec(vec![x1, x2], vec![2]).unwrap());
//     }
//     data
// }

// fn generate_image_data(n_samples: usize) -> Result<Vec<Tensor>, AstraError> {
//     let mut rng = rand::thread_rng();
//     let mut data: Vec<Tensor> = Vec::with_capacity(n_samples);

//     for _ in 0..n_samples / 2 {
//         data.push(Tensor::from_fn(vec![3, 4, 4], || rng.gen_range(0.0..0.5)));
//     }
//     for _ in (n_samples / 2)..n_samples {
//         data.push(Tensor::from_fn(vec![3, 4, 4], || rng.gen_range(3.0..3.5)));
//     }

//     Ok(data)
// }

// fn test_image_rec() -> Result<(), Box<dyn Error>> {
//     let ns = 500;

//     let data = generate_image_data(ns)?;
//     let mut targets: Vec<Tensor> = Vec::with_capacity(ns);
//     for _ in 0..ns / 2 {
//         targets.push(Tensor::from_vec(vec![1., 0.], vec![2])?);
//     }
//     for _ in ns / 2..ns {
//         targets.push(Tensor::from_vec(vec![0., 1.], vec![2])?);
//     }

//     let s1 = data[0].clone();
//     let s2 = data[ns / 2 + 1].clone();

//     let conv_layer = LayerConv2D::new(
//         vec![4, 4],
//         vec![2, 2],
//         3,
//         2,
//         Padding::Valid,
//         (1, 1),
//         Box::new(LeakyReLU::new(0.3)),
//     );
//     let flat_layer = LayerFlatten::new();
//     let hidden_layer = LayerDense::new(6, 18, Box::new(LeakyReLU::new(0.3)))?;
//     let output_layer = LayerDense::new(2, 6, Box::new(Softmax::new()))?;

//     let mut net = Net::new(Box::new(CategoricalCrossEntropy::new()), 0.0001, Some(3.0));

//     net.add_layer(Box::new(conv_layer));
//     net.add_layer(Box::new(flat_layer));
//     net.add_layer(Box::new(hidden_layer));
//     net.add_layer(Box::new(output_layer));

//     let r1 = net.feed_forward(&s1)?;
//     let r2 = net.feed_forward(&s2)?;

//     println!("r1 pretrain = {:#?}", r1);
//     println!("r2 pretrain = {:#?}", r2);

//     for (inp, tar) in data.into_iter().zip(targets.into_iter()) {
//         net.back_propagation(&inp, &tar)?;
//     }

//     let r1 = net.feed_forward(&s1)?;
//     let r2 = net.feed_forward(&s2)?;

//     println!("r1 posttrain = {:#?}", r1);
//     println!("r2 posttrain = {:#?}", r2);

//     Ok(())
// }

mod error;
mod helper_traits;
mod tensor2;
use crate::error::AstraError;
use std::error::Error;
use tensor2::Tensor;

fn main() -> Result<(), Box<dyn Error>> {
    let tensor = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], vec![3, 3]).unwrap();
    let sliced = tensor.slice(&[(1, 2), (0, 1)]).unwrap();
    let expected = Tensor::from_vec(vec![4, 5, 7, 8], vec![2, 2]).unwrap();

    println!("{:#?}", tensor);
    println!("{:#?}", sliced);
    println!("{:#?}", expected);

    Ok(())
}
