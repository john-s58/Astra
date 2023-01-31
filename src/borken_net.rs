use nalgebra::DMatrix;
use rand::Rng;
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};

#[derive(PartialEq, Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Net {
    pub config: Vec<usize>,
    pub weights: Vec<DMatrix<f32>>,
    pub biases: Vec<Vec<f32>>,
    pub activation: ActivationFunction,
    pub mutation_rate: Option<f32>,
}

impl Net {

    pub fn from_config(config: Vec<usize>, activation: ActivationFunction) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            config: config.clone(),
            weights: config
                .iter()
                .zip(config.iter().skip(1))
                .map(|(&cur, &last)| {
                    // DMatrix::from_fn(last, curr + 1, |_, _| gen_range(-1., 1.))
                    DMatrix::<f32>::from_distribution(last, cur, &StandardNormal, &mut rng)
                        * (2. / last as f32).sqrt()
                })
                .collect(),
            biases: config
                .iter()
                .map(|&l_sz| (0..l_sz).map(|_| rng.gen_range(0.0..1.0)).collect())
                .collect(),
            activation,
            mutation_rate: None,
        }
    }

    pub fn with_mutation(mut self, mutation_rate: f32) -> Self {
        self.mutation_rate = Some(mutation_rate);
        self
    }

    pub fn crossover(a: &Net, b: &Net) -> Self {
        assert_eq!(a.config, b.config, "NN configs not same.");
        let mut rng = rand::thread_rng();
        Self {
            config: a.config.to_owned(),
            activation: a.activation,
            mutation_rate: a.mutation_rate,
            weights: a
                .weights
                .iter()
                .zip(b.weights.iter())
                .map(|(m1, m2)| {
                    m1.zip_map(m2, |ele1, ele2| {
                        if rng.gen_range(0.0..1.0) < 0.5 {
                            ele1
                        } else {
                            ele2
                        }
                    })
                })
                .collect(),
            biases: a
                .clone()
                .biases
                .into_iter()
                .zip(b.clone().biases.into_iter())
                .map(|(b1, b2)| {
                    b1.into_iter()
                        .zip(b2.into_iter())
                        .map(|(ele1, ele2)| {
                            if rng.gen_range(0.0..1.0) < 0.5 {
                                ele1
                            } else {
                                ele2
                            }
                        })
                        .collect()
                })
                .collect(),
        }
    }

    pub fn mutate(&mut self) {
        match self.mutation_rate {
            Some(m) => {
                let mut rng = rand::thread_rng();
                for layer in &mut self.weights {
                    for weight in layer {
                        if rng.gen_range(0.0..1.0) < m {
                            *weight = rng.sample::<f32, StandardNormal>(StandardNormal);
                        }
                    }
                }
            }
            None => {}
        }
    }

    pub fn feed_forward(&self, inputs: &Vec<f32>) -> Vec<f32> {
        // println!("inputs: {:?}", inputs);
        let mut y = DMatrix::from_vec(inputs.len(), 1, inputs.to_vec());

        println!("feed_forward 1");

        for i in 0..self.config.len() - 1 {
            y = (&self.weights[i] * y.insert_row(self.config[i] - 1, 1.)
                + DMatrix::from_vec(self.biases[i].len(), 1, self.biases[i].to_vec()))
            .map(|x| match self.activation {
                ActivationFunction::ReLU => x.max(0.),
                ActivationFunction::Sigmoid => 1. / (1. + (-x).exp()),
                ActivationFunction::Tanh => x.tanh(),
            });
            
        }
        y.column(0).data.into_slice().to_vec()
    }
}

impl Net {
    /*

    This function takes in four parameters:
    - `inputs`: a `Vec` of input vectors for the neural network.
    - `targets`: a `Vec` of target vectors for the neural network.
    - `learning_rate`: a `f32` representing the learning rate of the network.
    - `epochs`: an `usize` representing the number of times the training data should be passed through the network.

    It uses a nested loop to iterate over the input and target vectors, and for each pair it:
    - Computes the output of the network using the `feed_forward` method.
    - Computes the error for each output using the difference between the target and the output.
    - Initializes matrices to store the weight and bias deltas.
    - Iterates over the layers of the network in reverse order, starting with the output layer.
    - Computes the weight and bias deltas for each layer using the errors and the outputs of the previous layers.
    - Adjusts the weights and biases of the network using the computed deltas and the learning rate.

    Finally, the function repeats this process for the number of specified `epochs`.

    Please note that this is an example implementation, and you may need to adjust it to suit the specifics of your problem and dataset.

        */
    pub fn train(
        &mut self,
        inputs: &Vec<Vec<f32>>,
        targets: &Vec<Vec<f32>>,
        learning_rate: f32,
        epochs: usize,
    ) {
        for _ in 0..epochs {
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let outputs = self.feed_forward(input);
                let mut errors = vec![0.0; *self.config.last().unwrap()];

                for (i, &output) in outputs.iter().enumerate() {
                    errors[i] = target[i] - output;
                }

                let mut weight_deltas = vec![DMatrix::zeros(0, 0); self.config.len() - 1];
                let mut bias_deltas = vec![vec![0.0; 0]; self.config.len() - 1];

                for (i, &size) in self.config.iter().enumerate().rev() {
                    if i == self.config.len() - 1 {
                        weight_deltas[i - 1] = DMatrix::from_vec(errors.len(), 1, errors.clone()).transpose();
                        bias_deltas[i - 1] = errors.clone();
                    } else {
                        let weights_i = self.weights[i].clone();
                        let errors_i = errors.clone();
                        errors = (weights_i.transpose() * DMatrix::from_vec(errors_i.len(), 1, errors_i.clone()).transpose())
                            .data
                            .as_slice()
                            .to_vec();

                        let outputs_i = self.feed_forward(input);
                        errors = errors
                            .into_iter()
                            .zip(outputs_i.into_iter())
                            .map(|(error, output)| {
                                error
                                    * match self.activation {
                                        ActivationFunction::ReLU => {
                                            if output > 0. {
                                                1.
                                            } else {
                                                0.
                                            }
                                        }
                                        ActivationFunction::Sigmoid => output * (1. - output),
                                        ActivationFunction::Tanh => 1. - output * output,
                                    }
                            })
                            .collect();

                        let ff = self.feed_forward(input).clone();
                        weight_deltas[i - 1] = DMatrix::from_vec(errors_i.len(), 1, errors_i.clone()).transpose()
                            * DMatrix::from_vec(ff.len(), 1, ff);
                        bias_deltas[i - 1] = errors_i;
                    }
                }

                for (i, &size) in self.config.iter().enumerate().rev() {
                    self.weights[i] += weight_deltas[i].clone() * learning_rate;
                    self.biases[i] = self.biases[i].clone()
                        .into_iter()
                        .zip(bias_deltas[i].clone().into_iter())
                        .map(|(bias, delta)| bias + delta * learning_rate)
                        .collect();
                }
            }
        }
    }


    fn train_single(&mut self, inputs: &Vec<f32>, targets: &Vec<f32>) {
        let outputs = self.feed_forward(inputs);

        assert_eq!(targets.len(), outputs.len());

        let mut errors = vec![0.0; *self.config.last().unwrap()];

        for (i, &output) in outputs.iter().enumerate() {
            errors[i] = targets[i] - output;
        }

        let mut weight_deltas = vec![DMatrix::zeros(0, 0); self.config.len() - 1];
        let mut bias_deltas = vec![vec![0.0; 0]; self.config.len() - 1];

        for (i, &layer_size) in self.config.iter().enumerate().rev() {
            if i == self.config.len() - 1 {
                weight_deltas[i - 1] = DMatrix::from_vec(errors.len(), 1, errors.clone()).transpose();
                bias_deltas[i - 1] = errors.clone();
            }
            else {
                let weights_i = self.weights[i].clone();
                        let errors_i = errors.clone();
                        errors = (weights_i.transpose() * DMatrix::from_vec(errors_i.len(), 1, errors_i.clone()).transpose())
                            .data
                            .as_slice()
                            .to_vec();

            }

        }

    }
}

fn main() {
    let test1 = Net::from_config(vec![4, 8, 8, 4], ActivationFunction::ReLU).with_mutation(0.15);

    let test2 = Net::from_config(vec![4, 8, 8, 4], ActivationFunction::ReLU).with_mutation(0.15);

    let mut test = Net::crossover(&test1, &test2);

   // println!("{:#?}", &test);

    let inp : Vec<Vec<f32>>= vec![
        vec![1.0, 1.0, 1.0, 1.0],
        vec![2.0, 2.0, 2.0, 2.0],
        vec![3.0, 3.0, 3.0, 3.0],
        vec![4.0, 4.0, 4.0, 4.0],
        vec![5.0, 5.0, 5.0, 5.0],
        vec![6.0, 6.0, 6.0, 6.0],
        vec![7.0, 7.0, 7.0, 7.0],
    ];

    let out: Vec<Vec<f32>> = vec![
        vec![1.0, 1.0, 1.0, 1.0],
        vec![2.0, 2.0, 2.0, 2.0],
        vec![3.0, 3.0, 3.0, 3.0],
        vec![4.0, 4.0, 4.0, 4.0],
        vec![5.0, 5.0, 5.0, 5.0],
        vec![6.0, 6.0, 6.0, 6.0],
        vec![7.0, 7.0, 7.0, 7.0],
    ];

    let sing_in =  vec![1.0f32, 1.0, 1.0, 1.0];
    let sing_out=  vec![1.0f32, 1.0, 1.0, 1.0];

    println!("{:?}", sing_in);
    println!("{:?}", test.config);

    test.train_single(&sing_in, &sing_out);

  //  println!("{:#?}", &test);


}
