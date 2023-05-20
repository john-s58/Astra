use crate::astra_net::activation::Activation;
use crate::astra_net::layer::Layer;
use crate::error::AstraError;
use crate::tensor::Tensor;

use rand::distributions::Uniform;
use rand::Rng;

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
        Self {
            filters: Self::xavier_init(&kernel_shape, n_channels, n_filters),
            kernel_shape,
            stride,
            padding,
            input_shape,
            activation,
            input: None,
            output: None,
        }
    }

    fn xavier_init(kernel_shape: &[usize], n_channels: usize, n_filters: usize) -> Tensor {
        let mut rng = rand::thread_rng();
        let scaling_factor = (6.0
            / (n_channels as f64 + n_filters as f64 + (kernel_shape[0] + kernel_shape[1]) as f64))
            .sqrt();
        let uniform_dist = Uniform::from(-scaling_factor..scaling_factor);
        Tensor::from_fn(
            vec![n_filters, n_channels, kernel_shape[0], kernel_shape[1]],
            || rng.sample(uniform_dist),
        )
    }

    fn convolution(&self, input: &Tensor, kernel: &Tensor) -> Result<Tensor, AstraError> {
        if input.ndim != 2 || kernel.ndim != 2 {
            return Err(AstraError::BadInputShape);
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
                    .pad(&[(self.padding, self.padding), (self.padding, self.padding)])?
            }
        }
        for y in 0..output_height {
            for x in 0..output_width {
                let (y_start, y_end) = (y * self.stride, y * self.stride + kernel_height - 1);
                let (x_start, x_end) = (x * self.stride, x * self.stride + kernel_width - 1);

                *(output.get_element_mut(&[y, x])?) = (padded_image
                    .slice(&[(y_start, y_end), (x_start, x_end)])?
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
    ) -> Result<Tensor, AstraError> {
        let input = match self.input.clone() {
            None => {
                return Err(AstraError::UninitializedLayerParameter(
                    "self.input".to_string(),
                ))
            }
            Some(x) => x,
        };

        let mut gradient = Tensor::zeros(&filter.shape);

        for c in 0..filter.shape[0] {
            let input_channel = input
                .slice(&[(c, c), (0, input.shape[1] - 1), (0, input.shape[2] - 1)])?
                .reshape(&[input.shape[1], input.shape[2]])?;

            let padded_filter_error = filter_error
                .to_owned()
                .reshape(&[filter_error.shape[1], filter_error.shape[2]])?
                .pad(&[
                    ((filter.shape[1] - 1) / 2, (filter.shape[1] - 1) / 2),
                    ((filter.shape[2] - 1) / 2, (filter.shape[2] - 1) / 2),
                ])?;

            for y in 0..filter.shape[1] {
                for x in 0..filter.shape[2] {
                    let a = gradient.get_element_mut(&[c, y, x])?;

                    let input_region = input_channel
                        .slice(&[(y, y + filter.shape[1] - 1), (x, x + filter.shape[2] - 1)])?;

                    let padded_filter_error_region = padded_filter_error
                        .slice(&[(y, y + filter.shape[1] - 1), (x, x + filter.shape[2] - 1)])?;

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
    ) -> Result<Tensor, AstraError> {
        let d_l_d_o =
            filter_error.to_owned() * self.activation.derive(layer_output_cur_filter.to_owned());

        let d_l_d_o = d_l_d_o
            .clone()
            .reshape(&[d_l_d_o.shape[1], d_l_d_o.shape[2]])?;

        let mut d_l_d_i = Tensor::zeros(&[
            filter.shape[0],
            layer_output_cur_filter.shape[0],
            layer_output_cur_filter.shape[1],
        ]);

        for c in 0..filter.shape[0] {
            let rotated_filter_channel = filter
                .slice(&[(c, c), (0, filter.shape[1] - 1), (0, filter.shape[2] - 1)])?
                .reshape(&[filter.shape[1], filter.shape[2]])?
                .rotate_180_degrees()?;

            let padded_d_l_d_o = d_l_d_o.pad(&[
                (filter.shape[1] - 1, filter.shape[1] - 1),
                (filter.shape[2] - 1, filter.shape[2] - 1),
            ])?;

            for y in 0..layer_output_cur_filter.shape[0] {
                for x in 0..layer_output_cur_filter.shape[1] {
                    let d_l_d_i_element = d_l_d_i.get_element_mut(&[c, y, x])?;

                    *d_l_d_i_element = (padded_d_l_d_o
                        .slice(&[(y, y + filter.shape[1] - 1), (x, x + filter.shape[2] - 1)])?
                        .reshape(&[filter.shape[1], filter.shape[2]])?
                        * rotated_filter_channel.clone())
                    .sum();
                }
            }
        }

        Ok(d_l_d_i)
    }
}

impl Layer for LayerConv2D {
    fn feed_forward(&mut self, inputs: &Tensor) -> Result<Tensor, AstraError> {
        self.input = Some(inputs.to_owned());

        let mut output: Vec<Tensor> = Vec::new();

        let (ks1, ks2) = (self.kernel_shape[0], self.kernel_shape[1]);

        for f_n in 0..self.filters.shape[0] {
            let mut cur_filter_result: Vec<Tensor> = Vec::with_capacity(self.filters.shape[1]);

            for c_n in 0..self.filters.shape[1] {
                let filter = self
                    .filters
                    .slice(&[(f_n, f_n), (c_n, c_n), (0, ks1 - 1), (0, ks2 - 1)])?
                    .reshape(&[ks1, ks2])?;

                let c_input = inputs
                    .slice(&[
                        (c_n, c_n),
                        (0, inputs.shape[1] - 1),
                        (0, inputs.shape[2] - 1),
                    ])?
                    .reshape(&[inputs.shape[1], inputs.shape[2]])?;

                cur_filter_result.push(self.convolution(&c_input, &filter)?);
            }
            let mut summed = Tensor::zeros(&cur_filter_result[0].shape);

            for t in cur_filter_result.into_iter() {
                summed = summed + t;
            }
            output.push(summed);
        }

        let layer_result = Tensor::stack(&output)?;
        self.output = Some(layer_result.clone());

        Ok(layer_result)
    }

    fn back_propagation(
        &mut self,
        error: Tensor,
        learning_rate: f64,
        clipping_value: Option<f64>,
    ) -> Result<Tensor, AstraError> {
        let mut gradients: Vec<Tensor> = Vec::new();
        let mut prev_layer_errors: Vec<Tensor> = Vec::new();

        let layer_output = self
            .output
            .clone()
            .ok_or(AstraError::UninitializedLayerParameter(
                "self.output".to_string(),
            ))?;

        for f_n in 0..self.filters.shape[0] {
            let cur_filter = self
                .filters
                .slice(&[
                    (f_n, f_n),
                    (0, self.filters.shape[1] - 1),
                    (0, self.filters.shape[2] - 1),
                    (0, self.filters.shape[3] - 1),
                ])?
                .reshape(&[
                    self.filters.shape[1],
                    self.filters.shape[2],
                    self.filters.shape[3],
                ])?;

            let cur_filter_error =
                error.slice(&[(f_n, f_n), (0, error.shape[1] - 1), (0, error.shape[2] - 1)])?;

            let layer_output_cur_filter = layer_output.slice(&[
                (f_n, f_n),
                (0, layer_output.shape[1] - 1),
                (0, layer_output.shape[2] - 1),
            ])?;

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
        let gradients_stacked = match clipping_value {
            None => Tensor::stack(&gradients)?,
            Some(v) => Tensor::stack(&gradients)?.clip(v),
        };
        self.filters = self.filters.to_owned() - (gradients_stacked * learning_rate);

        // Sum errors for the previous layer
        let prev_layer_error_sum = prev_layer_errors
            .into_iter()
            .reduce(|a, b| a + b)
            .ok_or(AstraError::CustomError("could not sum errors".to_string()))?;

        Ok(prev_layer_error_sum)
    }
}
