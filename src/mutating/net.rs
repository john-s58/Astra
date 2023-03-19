use crate::astra_net::activation::{Activation, LeakyReLU, Softmax};
use crate::astra_net::layer::LayerDense;
use crate::astra_net::Net;

use crate::tensor::Tensor;

use ndarray_rand::rand_distr::{Distribution, Normal};
use rand::Rng;

pub struct MutatingNet {
    config: Vec<usize>,
    weights: Vec<Tensor>,
    biases: Vec<Tensor>,
    activation: Box<dyn Activation>,
    mutation_rate: f64,
}

impl MutatingNet {
    pub fn from_config(config: Vec<usize>) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        Self {
            config: config.clone(),
            weights: config
                .clone()
                .into_iter()
                .zip(config.clone().into_iter().skip(1))
                .map(|(prev, cur)| {
                    Tensor::from_vec(
                        (0..cur * prev).map(|_| normal.sample(&mut rng)).collect(),
                        vec![prev, cur],
                    ) * (2.0 / (prev + cur) as f64).sqrt()
                })
                .collect(),
            biases: config
                .into_iter()
                .skip(1)
                .map(|sz| Tensor::from_element(0.0, vec![sz]))
                .collect(),
            activation: Box::new(LeakyReLU::new(0.3)),
            mutation_rate: 0.15,
        }
    }

    pub fn feed_forward(&self, input: &Tensor) -> Tensor {
        let mut output = input.to_owned().reshape(vec![1, input.len()]).unwrap();

        for (weight, bias) in self
            .weights
            .to_owned()
            .into_iter()
            .zip(self.biases.to_owned().into_iter())
        {
            output =
                output.dot(&weight).unwrap() + bias.clone().reshape(vec![1, bias.len()]).unwrap();

            output = self.activation.call(output)
        }
        output
    }

    pub fn crossover(mut self, right: &Self) -> Option<Self> {
        if self.config.len() != right.config.len() {
            return None;
        }
        for (d_r, d_l) in self.config.iter().zip(right.config.iter()) {
            if d_r != d_l {
                return None;
            }
        }
        let mut rng = rand::thread_rng();

        self.weights = self
            .weights
            .into_iter()
            .zip(right.weights.to_owned().into_iter())
            .map(|(weights_self, weights_right)| {
                Tensor::from_vec(
                    weights_self
                        .clone()
                        .into_iter()
                        .zip(weights_right.into_iter())
                        .map(|(w_l, w_r)| {
                            if rng.gen_range(0.0..1.0) < 0.5 {
                                w_l
                            } else {
                                w_r
                            }
                        })
                        .collect(),
                    weights_self.shape,
                )
            })
            .collect();
        self.biases = self
            .biases
            .into_iter()
            .zip(right.biases.to_owned().into_iter())
            .map(|(weights_self, weights_right)| {
                Tensor::from_vec(
                    weights_self
                        .clone()
                        .into_iter()
                        .zip(weights_right.into_iter())
                        .map(|(w_l, w_r)| {
                            if rng.gen_range(0.0..1.0) < 0.5 {
                                w_l
                            } else {
                                w_r
                            }
                        })
                        .collect(),
                    weights_self.shape,
                )
            })
            .collect();

        Some(self)
    }

    pub fn mutate(&mut self) {
        //self.weights = self.weights.iter().map(f)
        todo!()
    }
}
