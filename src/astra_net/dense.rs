use crate::astra_net::activation::Activation;
use crate::astra_net::layer::Layer;
use crate::astra_net::net_error::NetError;
use crate::tensor::Tensor;
use ndarray_rand::rand_distr::{Distribution, Normal};

pub struct LayerDense {
    weights: Tensor,
    biases: Tensor,
    activation: Box<dyn Activation>,
    input: Option<Tensor>,
    output: Option<Tensor>,
}
impl LayerDense {
    pub fn new(
        size: usize,
        input_size: usize,
        activation: Box<dyn Activation>,
    ) -> Result<Self, NetError> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(-1.0, 1.0).unwrap();

        Ok(Self {
            weights: Tensor::from_vec(
                (0..size * input_size)
                    .map(|_| normal.sample(&mut rng))
                    .collect(),
                vec![input_size, size],
            )
            .map_err(NetError::TensorBasedError)?
                * (2.0 / (input_size + size) as f64).sqrt(),
            biases: Tensor::from_element(0.0, vec![size]),
            activation,
            input: None,
            output: None,
        })
    }
}

impl Layer for LayerDense {
    fn feed_forward(&mut self, inputs: &Tensor) -> Result<Tensor, NetError> {
        self.input = Some(inputs.clone());

        if inputs.len() != self.weights.shape[0] {
            return Err(NetError::BadInputShape);
        }

        let inputs_mat = inputs
            .to_owned()
            .reshape(&[1, inputs.len()])
            .map_err(NetError::TensorBasedError)?;

        self.output = Some(
            inputs_mat
                .dot(&self.weights)
                .map_err(NetError::TensorBasedError)?
                + self
                    .biases
                    .clone()
                    .reshape(&[1, self.biases.len()])
                    .map_err(NetError::TensorBasedError)?,
        );

        self.output = Some(self.activation.call(self.output.clone().unwrap()));

        Ok(self.output.clone().unwrap())
    }

    fn back_propagation(&mut self, error: Tensor, learning_rate: f64) -> Result<Tensor, NetError> {
        let delta_output = self.activation.derive(self.output.clone().unwrap());

        let err = error
            .clone()
            .reshape(&[1, error.len()])
            .map_err(NetError::TensorBasedError)?
            * delta_output;

        let input_mat = self
            .input
            .clone()
            .ok_or(NetError::UninitializedLayerParameter(
                "self.input".to_string(),
            ))?
            .reshape(&[1, self.input.clone().unwrap().len()])
            .map_err(NetError::TensorBasedError)?;

        let err_mat = err
            .clone()
            .reshape(&[1, err.len()])
            .map_err(NetError::TensorBasedError)?;

        let delta_weights = input_mat
            .transpose()
            .map_err(NetError::TensorBasedError)?
            .dot(&err_mat)
            .map_err(NetError::TensorBasedError)?;
        let delta_biases = err.sum() / err.len() as f64;

        self.weights = self.weights.clone() - (delta_weights * learning_rate);
        self.biases = self.biases.clone() - (delta_biases * learning_rate);

        err.dot(
            &self
                .weights
                .transpose()
                .map_err(NetError::TensorBasedError)?,
        )
        .map_err(NetError::TensorBasedError)
    }
}
