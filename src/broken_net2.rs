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

    pub fn back_propagation(&mut self, inputs: &Vec<f32>, targets: &Vec<f32>, learning_rate: f32) { 
        // Consider &mut self as param and change original net
        let output_vec = self.feed_forward(inputs);
        let output = DMatrix::from_vec(output_vec.len(), 1, output_vec);
        let mut error = DMatrix::from_vec(targets.len(), 1, targets.to_vec()) - output;

        for ((layer_weights,
             layer_biases), 
             layer_activation) in 
             self.weights.iter_mut()
            .zip(self.biases.iter_mut())
            .zip(self.activation.iter())
            .rev(){
                let error_deactiv = match layer_activation {
                    ActivationFunction::ReLU => {error.clone().map(|x| if x > 0.0 {1.0} else {0.0})},
                    _ => error.clone(),
                };
              error = error.component_mul(&error_deactiv);
              println!("{:#?}", &error);
              println!("{:#?}", &layer_weights);

                let delta_weights = error.transpose() * layer_weights.clone();
                let delta_biases = error.row_sum() / error.nrows() as f32;

                //*layer_weights -= (delta_weights * learning_rate);
                //*layer_biases = (DMatrix::from_vec(layer_biases.len(), 1, layer_biases.to_vec()) - (delta_biases * learning_rate)).data.into();
                //error = error * layer_weights.transpose();

        }

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
    let mut net = Net::new()
        .add_layer(3, ActivationFunction::InputLayer)
        .add_layer(4, ActivationFunction::ReLU)
        .add_layer(4, ActivationFunction::ReLU)
        .add_layer(3, ActivationFunction::Sigmoid)
        .with_mutation(0.15);

    let input: Vec<f32> = vec![1.2, 3.4, 5.6];
    let targets = vec![2.4, 6.8, 11.2];

    let mut outs = net.feed_forward(&input);
    //println!("{:#?}", &outs);

    for _ in 0..10 {
        net.back_propagation(&input, &targets, 0.1);
        outs = net.feed_forward(&input);
        //println!("{:#?}", &outs);
    }


    //println!("{:#?}", outs);

}