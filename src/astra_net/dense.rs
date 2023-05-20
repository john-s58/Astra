use crate::astra_net::activation::Activation;
use crate::astra_net::layer::Layer;
use crate::error::AstraError;
use crate::tensor::Tensor;
use ndarray_rand::rand_distr::{Distribution, Normal};

pub struct LayerDense {
    weights: Tensor,
    biases: Tensor,
    activation: Box<dyn Activation>,
    input: Option<Tensor>,
    z: Option<Tensor>,
    output: Option<Tensor>,
}
impl LayerDense {
    pub fn new(
        size: usize,
        input_size: usize,
        activation: Box<dyn Activation>,
    ) -> Result<Self, AstraError> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(-1.0, 1.0).unwrap();

        Ok(Self {
            weights: Tensor::from_vec(
                (0..size * input_size)
                    .map(|_| normal.sample(&mut rng))
                    .collect(),
                vec![input_size, size],
            )? * (2.0 / (input_size + size) as f64).sqrt(),
            biases: Tensor::from_element(0.0, vec![size]),
            activation,
            input: None,
            z: None,
            output: None,
        })
    }
}

impl Layer for LayerDense {
    fn feed_forward(&mut self, inputs: &Tensor) -> Result<Tensor, AstraError> {
        if inputs.len() != self.weights.shape[0] {
            return Err(AstraError::BadInputShape);
        }

        let inputs_mat = inputs.to_owned().reshape(&[1, inputs.len()])?;
        self.input = Some(inputs_mat.clone());

        self.z = Some(
            inputs_mat.dot(&self.weights)?
                + self.biases.clone().reshape(&[1, self.biases.len()])?,
        );

        self.output = Some(self.activation.call(self.z.clone().unwrap()));

        Ok(self.output.clone().unwrap())
    }

    fn back_propagation(
        &mut self,
        output_gradient: Tensor,
        learning_rate: f64,
        clipping_value: Option<f64>,
    ) -> Result<Tensor, AstraError> {
        let da_dz: Tensor = self.activation.derive(self.z.clone().ok_or(
            AstraError::UninitializedLayerParameter("self.z".to_string()),
        )?);

        let dl_da_da_dz = output_gradient
            .clone()
            .reshape(&[1, output_gradient.len()])?
            * da_dz;

        let input_mat = self
            .input
            .clone()
            .ok_or(AstraError::UninitializedLayerParameter(
                "self.input".to_string(),
            ))?;

        let delta_weights = match clipping_value {
            None => input_mat.transpose()?.dot(&dl_da_da_dz)?,
            Some(v) => input_mat.transpose()?.dot(&dl_da_da_dz)?.clip(v),
        };

        let delta_biases = dl_da_da_dz.sum() / dl_da_da_dz.len() as f64;

        self.weights = self.weights.clone() - (delta_weights * learning_rate);
        self.biases = self.biases.clone() - (delta_biases * learning_rate);

        dl_da_da_dz.dot(&self.weights.transpose()?)
    }
}
