use nalgebra::DMatrix;
use rand::Rng;
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};

#[derive(PartialEq, Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationFunction {
    InputLayer,
    ReLU,
    Sigmoid,
    Tanh,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Net {
    pub config: Vec<usize>,
    pub weights: Vec<DMatrix<f32>>,
    pub biases: Vec<Vec<f32>>,
    pub activation: Vec<ActivationFunction>,
    pub mutation_rate: Option<f32>,
}
// CONSIDER BIAS FOR LAYER AND NOT ONLY FOR EACH NEURON
impl Net {
    pub fn new() -> Self{
        Self {
            config: Vec::<usize>::new(),
            weights: Vec::<DMatrix<f32>>::new(),
            biases: Vec::<Vec<f32>>::new(),
            activation: Vec::<ActivationFunction>::new(),
            mutation_rate: None,
        }
    }

    pub fn add_layer(mut self, layer_size: usize, layer_activation: ActivationFunction) -> Self{
        if self.config.len() == 0 {
            match layer_activation {
                ActivationFunction::InputLayer => {
                    self.config.push(layer_size);
                    self.activation.push(layer_activation);
                    return self
                },
                _ => panic!("Input Layer Activation Must Be InputLayer")
            }
        }
        else {
            let mut rng = rand::thread_rng();

            // Hat El Initialization
            self.weights.push(DMatrix::<f32>::from_distribution(*self.config.last().unwrap(), 
            layer_size, &StandardNormal, &mut rng)
            * (2. / *self.config.last().unwrap() as f32).sqrt());
            self.biases.push(
                (0..layer_size).map(|_| rng.gen_range(0.0..1.0)).collect()
            );
            self.config.push(layer_size);
            self.activation.push(layer_activation);

            return self
        }
    }

    pub fn with_mutation(mut self, mutation_rate: f32) -> Self {
        self.mutation_rate = Some(mutation_rate);
        self
    }

    pub fn feed_forward(&self, inputs: &Vec<f32>) -> Vec<f32> {
        let mut outputs =  DMatrix::from_vec(inputs.len(), 1, inputs.to_vec());

        for ((w, b), 
                    activ) in self.weights.iter()
                                                    .zip(self.biases.iter())
                                                    .zip(self.activation.iter().skip(1)){
            outputs = w.transpose() * outputs +
             DMatrix::from_vec(b.len(), 1, b.to_vec());
            match activ {
                ActivationFunction::ReLU => {outputs = outputs.map(|x| if x > 0.0 {x} else {0.0})},
                ActivationFunction::Sigmoid => {outputs = outputs.map(|x| 1.0 / (1.0 + (-x).exp()))},
                ActivationFunction::Tanh => {outputs = outputs.map(|x| x.tanh())},
                _ => ()
            }
        }
        outputs.data.into()
    }

    pub fn back_propagation(self, inputs: &Vec<f32>, targets: &Vec<f32>) -> Self { 
        // Consider &mut self as param and change original net
        self
    }

    pub fn crossover(a: &Net, b: &Net) -> Self {
        assert_eq!(a.config, b.config, "The 2 Networks Structure Is Not The Same");
        let mut rng = rand::thread_rng();
        Self {
            config: a.config.to_owned(),
            activation: a.activation.to_owned(),
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

}

fn main(){
    let net = Net::new()
        .add_layer(3, ActivationFunction::InputLayer)
        .add_layer(4, ActivationFunction::ReLU)
        .add_layer(4, ActivationFunction::ReLU)
        .add_layer(3, ActivationFunction::Sigmoid)
        .with_mutation(0.15);

    let input: Vec<f32> = vec![1.2, 3.4, 5.6];

    //println!("{:#?}", net);

    let outs = net.feed_forward(&input);

    println!("{:#?}", outs);

}