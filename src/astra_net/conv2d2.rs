use crate::astra_net::activation::Activation;
use crate::astra_net::layer::Layer;
use crate::astra_net::net_error::NetError;
use crate::tensor::Tensor;

use ndarray_rand::rand_distr::{Distribution, Normal};

pub struct LayerConv2D {
    pub filters: Tensor,
    pub kernel_shape: Vec<usize>,
    pub stride: usize,
    pub padding: usize,
    pub input_shape: Vec<usize>,
    pub activation: Box<dyn Activation>,
    pub input: Option<Tensor>,
    pub output: Option<Tensor>,
}

impl LayerConv2D {
    pub fn new(
        input_shape: Vec<usize>,
        kernel_shape: Vec<usize>,
        n_channels: usize,
        n_filters: usize,
        padding: usize,
        stride: usize,
        activation: Box<dyn Activation>,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0., 1.0).unwrap();
        Self {
            filters: Tensor::from_fn(
                vec![n_filters, n_channels, kernel_shape[0], kernel_shape[1]],
                || normal.sample(&mut rng),
            ),
            kernel_shape,
            stride,
            padding,
            input_shape,
            activation,
            input: None,
            output: None,
        }
    }

    fn convolution(&self, input: &Tensor, kernel: &Tensor) -> Result<Tensor, NetError> {
        if input.ndim != 2 || kernel.ndim != 2 {
            return Err(NetError::BadInputShape);
        }

        let (img_height, img_width) = (input.shape[0], input.shape[1]);
        let (kernel_height, kernel_width) = (kernel.shape[0], kernel.shape[1]);

        let output_height = (img_height - kernel_height + 2 * self.padding) / self.stride + 1;
        let output_width = (img_width - kernel_width + 2 * self.padding) / self.stride + 1;

        let mut output = Tensor::zeros(&[output_height, output_width]);

        let mut padded_image = input.to_owned();

        match self.padding {
            0 => {}
            _ => {
                padded_image = padded_image
                    .pad(&[(self.padding, self.padding), (self.padding, self.padding)])
                    .map_err(NetError::TensorBasedError)?
            }
        }
        for y in 0..output_height {
            for x in 0..output_width {
                let (y_start, y_end) = (y * self.stride, y * self.stride + kernel_height - 1);
                let (x_start, x_end) = (x * self.stride, x * self.stride + kernel_width - 1);

                *(output
                    .get_element_mut(&[y, x])
                    .map_err(NetError::TensorBasedError)?) = (padded_image
                    .slice(&[(y_start, y_end), (x_start, x_end)])
                    .map_err(NetError::TensorBasedError)?
                    * kernel.to_owned())
                .sum();
            }
        }
        Ok(self.activation.call(output))
    }

    fn calculate_gradients_per_filter(
        &self,
        filter_error: &Tensor,
        filter: &Tensor,
    ) -> Result<Tensor, NetError> {
        let input = match self.input.clone() {
            None => return Err(NetError::CustomError("uninit".to_string())),
            Some(x) => x,
        };

        let mut gradient = Tensor::zeros(&filter.shape);

        for c in 0..filter.shape[0] {
            let input_channel = input
                .slice(&[(c, c), (0, input.shape[1] - 1), (0, input.shape[2] - 1)])
                .map_err(NetError::TensorBasedError)?
                .reshape(&[input.shape[1], input.shape[2]])
                .map_err(NetError::TensorBasedError)?;

            let padded_filter_error = filter_error
                .to_owned()
                .reshape(&[filter_error.shape[1], filter_error.shape[2]])
                .map_err(NetError::TensorBasedError)?
                .pad(&[
                    ((filter.shape[1] - 1) / 2, (filter.shape[1] - 1) / 2),
                    ((filter.shape[2] - 1) / 2, (filter.shape[2] - 1) / 2),
                ])
                .map_err(NetError::TensorBasedError)?;

            for y in 0..filter.shape[1] {
                for x in 0..filter.shape[2] {
                    let a = gradient
                        .get_element_mut(&[c, y, x])
                        .map_err(NetError::TensorBasedError)?;

                    let input_region = input_channel
                        .slice(&[(y, y + filter.shape[1] - 1), (x, x + filter.shape[2] - 1)])
                        .map_err(NetError::TensorBasedError)?;

                    let padded_filter_error_region = padded_filter_error
                        .slice(&[(y, y + filter.shape[1] - 1), (x, x + filter.shape[2] - 1)])
                        .map_err(NetError::TensorBasedError)?;

                    *a = (input_region * padded_filter_error_region).sum();
                }
            }
        }

        Ok(gradient)
    }

    fn calculate_error_to_previous_layer_per_filter(
        &self,
        filter_error: &Tensor,
        filter: &Tensor,
        layer_output_cur_filter: &Tensor,
    ) -> Result<Tensor, NetError> {
        let dL_dO =
            filter_error.to_owned() * self.activation.derive(layer_output_cur_filter.to_owned());

        let dL_dO = dL_dO
            .clone()
            .reshape(&[dL_dO.shape[1], dL_dO.shape[2]])
            .map_err(NetError::TensorBasedError)?;

        let mut dL_dI = Tensor::zeros(&[
            filter.shape[0],
            layer_output_cur_filter.shape[0],
            layer_output_cur_filter.shape[1],
        ]);

        for c in 0..filter.shape[0] {
            let rotated_filter_channel = filter
                .slice(&[(c, c), (0, filter.shape[1] - 1), (0, filter.shape[2] - 1)])
                .map_err(NetError::TensorBasedError)?
                .reshape(&[filter.shape[1], filter.shape[2]])
                .map_err(NetError::TensorBasedError)?
                .rotate_180_degrees()
                .map_err(NetError::TensorBasedError)?;

            let padded_dL_dO = dL_dO
                .pad(&[
                    (filter.shape[1] - 1, filter.shape[1] - 1),
                    (filter.shape[2] - 1, filter.shape[2] - 1),
                ])
                .map_err(NetError::TensorBasedError)?;

            for y in 0..layer_output_cur_filter.shape[0] {
                for x in 0..layer_output_cur_filter.shape[1] {
                    let dL_dI_element = dL_dI
                        .get_element_mut(&[c, y, x])
                        .map_err(NetError::TensorBasedError)?;

                    *dL_dI_element = (padded_dL_dO
                        .slice(&[(y, y + filter.shape[1] - 1), (x, x + filter.shape[2] - 1)])
                        .map_err(NetError::TensorBasedError)?
                        .reshape(&[filter.shape[1], filter.shape[2]])
                        .map_err(NetError::TensorBasedError)?
                        * rotated_filter_channel.clone())
                    .sum();
                }
            }
        }

        Ok(dL_dI)
    }
}

impl Layer for LayerConv2D {
    fn feed_forward(&mut self, inputs: &Tensor) -> Result<Tensor, NetError> {
        self.input = Some(inputs.to_owned());

        let mut output: Vec<Tensor> = Vec::new();

        let (ks1, ks2) = (self.kernel_shape[0], self.kernel_shape[1]);

        for f_n in 0..self.filters.shape[0] {
            let mut cur_filter_result: Vec<Tensor> = Vec::with_capacity(self.filters.shape[1]);

            for c_n in 0..self.filters.shape[1] {
                let filter = self
                    .filters
                    .slice(&[(f_n, f_n), (c_n, c_n), (0, ks1 - 1), (0, ks2 - 1)])
                    .map_err(NetError::TensorBasedError)?
                    .reshape(&[ks1, ks2])
                    .map_err(NetError::TensorBasedError)?;

                let c_input = inputs
                    .slice(&[
                        (c_n, c_n),
                        (0, inputs.shape[1] - 1),
                        (0, inputs.shape[2] - 1),
                    ])
                    .map_err(NetError::TensorBasedError)?
                    .reshape(&[inputs.shape[1], inputs.shape[2]])
                    .map_err(NetError::TensorBasedError)?;

                cur_filter_result.push(self.convolution(&c_input, &filter)?);
            }
            let mut summed = Tensor::zeros(&cur_filter_result[0].shape);

            for t in cur_filter_result.into_iter() {
                summed = summed + t;
            }
            output.push(summed);
        }

        let layer_result = Tensor::stack(&output).map_err(NetError::TensorBasedError)?;
        self.output = Some(layer_result.clone());

        Ok(layer_result)
    }

    fn back_propagation(&mut self, error: Tensor, learning_rate: f64) -> Result<Tensor, NetError> {
        let mut gradients: Vec<Tensor> = Vec::new();
        let mut prev_layer_errors: Vec<Tensor> = Vec::new();

        let layer_output = self.output.clone().ok_or(NetError::CustomError(
            "Layer Output not initialized".to_string(),
        ))?;

        for f_n in 0..self.filters.shape[0] {
            let cur_filter = self
                .filters
                .slice(&[
                    (f_n, f_n),
                    (0, self.filters.shape[1] - 1),
                    (0, self.filters.shape[2] - 1),
                    (0, self.filters.shape[3] - 1),
                ])
                .map_err(NetError::TensorBasedError)?
                .reshape(&[
                    self.filters.shape[1],
                    self.filters.shape[2],
                    self.filters.shape[3],
                ])
                .map_err(NetError::TensorBasedError)?;

            let cur_filter_error = error
                .slice(&[(f_n, f_n), (0, error.shape[1] - 1), (0, error.shape[2] - 1)])
                .map_err(NetError::TensorBasedError)?;

            let layer_output_cur_filter = layer_output
                .slice(&[
                    (f_n, f_n),
                    (0, layer_output.shape[1] - 1),
                    (0, layer_output.shape[2] - 1),
                ])
                .map_err(NetError::TensorBasedError)?;

            let gradient = self.calculate_gradients_per_filter(&cur_filter_error, &cur_filter)?;
            gradients.push(gradient.clone());

            let prev_layer_error = self.calculate_error_to_previous_layer_per_filter(
                &cur_filter_error,
                &cur_filter,
                &layer_output_cur_filter,
            )?;
            prev_layer_errors.push(prev_layer_error);
        }

        // Update filter weights
        self.filters = self.filters.to_owned()
            - (Tensor::stack(&gradients).map_err(NetError::TensorBasedError)? * learning_rate);

        // Sum errors for the previous layer
        let prev_layer_error_sum = prev_layer_errors
            .into_iter()
            .reduce(|a, b| a + b)
            .ok_or(NetError::CustomError("could not sum errors".to_string()))?;

        Ok(prev_layer_error_sum)
    }
}
