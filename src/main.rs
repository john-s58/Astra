use crate::astra_net::activation::{LeakyReLU, Softmax};
use crate::astra_net::dense::LayerDense;
use crate::astra_net::Net;
use crate::astra_net::layer::Layer;
use crate::mutating::net::MutatingNet;
use crate::tensor::Tensor;
use crate::astra_net::conv2d::LayerConv2D;

mod astra_net;
mod mutating;
mod tensor;
use rand::Rng;

fn main() {
    let mut conv = LayerConv2D {
        filters: vec![
            Tensor::matrix(2, 2, vec![1.0,1.0,1.0,1.0]).unwrap(),
            Tensor::matrix(2, 2, vec![2.0,1.0,2.0,1.0]).unwrap()
        ],
        kernal_shape: vec![2,2],
        stride: 1,
        padding: 0,
        input_shape: vec![5,5],
        activation: Box::new(LeakyReLU::new(0.3)),
        input: None,
        output: None
    };

    let conv_res = conv.feed_forward(&Tensor::from_element(5.0, vec![5,5]));

    println!("conv_res {:#?}", conv_res);

   
    // test_tensors();
    // test_astra_net_tensor();
    // test_astra_mutating_mutatingnet();

}


fn test_tensors() {
    let ten = Tensor::from_element(1.0, vec![3, 3]);
    let sten = ten.get_sub_matrix(&[1, 1], &[2, 2]).unwrap();

    let slten = ten.slice(&[(0, 2), (0, 2)]).unwrap();

    println!("sten : \n{:#?}", sten);
    println!("slten : \n{:#?}", slten);
}

fn test_astra_net_tensor() {
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

    let input_data = generate_2d_cluster_dataset(50000);

    let target_data: Vec<Tensor> = (0..input_data.len())
        .map(|x| {
            if x < input_data.len() / 2 {
                Tensor::from_vec(vec![1.0, 0.0], vec![2])
            } else {
                Tensor::from_vec(vec![0.0, 1.0], vec![2])
            }
        })
        .collect();

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

fn test_astra_mutating_mutatingnet() {
    let net = MutatingNet::from_config(vec![3, 5, 3]);
    let net2 = MutatingNet::from_config(vec![3, 5, 3]);

    let mut net_combined = net.crossover(&net2).unwrap();

    net_combined.mutate();

    let input = Tensor::from_element(1.3, vec![3]);

    let res = net_combined.feed_forward(&input);

    println!("mut net feed forward result = {:#?}", res);
}
