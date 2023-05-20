use crate::astra_net::activation::Activation;
use crate::astra_net::layer::Layer;
use crate::error::AstraError;
use crate::tensor::Tensor;

use rand::distributions::Uniform;
use rand::Rng;

#[derive(Debug, Clone, Copy)]
pub enum Padding {
    Same,
    Valid,
    Custom((usize, usize), (usize, usize)),
}

pub struct LayerConv2D {
    pub filters: Tensor,
    pub filter_shape: Vec<usize>,
    pub stride: (usize, usize),
    pub padding: Padding,
    pub input_shape: Vec<usize>,
    pub activation: Box<dyn Activation>,
    pub input: Option<Tensor>,
    pub output: Option<Tensor>,
    pub z: Option<Tensor>,
}

impl LayerConv2D {
    pub fn new(
        input_shape: Vec<usize>,
        filter_shape: Vec<usize>,
        n_channels: usize,
        n_filters: usize,
        padding: Padding,
        stride: (usize, usize),
        activation: Box<dyn Activation>,
    ) -> Self {
        Self {
            filters: Self::xavier_init(&filter_shape, n_channels, n_filters),
            filter_shape,
            stride,
            padding,
            input_shape,
            activation,
            input: None,
            output: None,
            z: None,
        }
    }

    fn xavier_init(filter_shape: &[usize], n_channels: usize, n_filters: usize) -> Tensor {
        let mut rng = rand::thread_rng();
        let scaling_factor = (6.0
            / (n_channels as f64 + n_filters as f64 + (filter_shape[0] + filter_shape[1]) as f64))
            .sqrt();
        let uniform_dist = Uniform::from(-scaling_factor..scaling_factor);
        Tensor::from_fn(
            vec![n_filters, n_channels, filter_shape[0], filter_shape[1]],
            || rng.sample(uniform_dist),
        )
    }

    fn cross_correlation(
        input: &Tensor,
        filter: &Tensor,
        stride: (usize, usize),
        padding: Padding,
    ) -> Result<Tensor, AstraError> {
        println!("is it here?");
        if input.ndim != 2 || filter.ndim != 2 {
            return Err(AstraError::UnsupportedDimension);
        }
        println!("not here");

        let (input_rows, input_cols) = (input.shape[0], input.shape[1]);
        let (filter_rows, filter_cols) = (filter.shape[0], filter.shape[1]);
        let (stride_rows, stride_cols) = stride;

        let mut padded_input = input.to_owned();

        let mut output_rows = (input_rows - filter_rows) / stride_rows + 1;
        let mut output_cols = (input_cols - filter_cols) / stride_cols + 1;

        match padding {
            Padding::Valid => {}
            Padding::Same => {
                (output_rows, output_cols) = (input_rows, input_cols);

                let mut padding_rows = (stride_rows - 1) * input_rows - stride_rows + filter_rows;
                match padding_rows % 2 {
                    0 => padding_rows = padding_rows / 2,
                    _ => padding_rows = padding_rows / 2 + 1,
                }
                match padding_rows % 2 {
                    0 => {
                        let padding_top = padding_rows / 2;
                        let padding_bottom = padding_rows - padding_top;
                    }
                    _ => {
                        let padding_top = padding_rows / 2 + 1;
                        let padding_bottom = padding_rows - padding_top;
                    }
                }
                let mut padding_cols = (stride_cols - 1) * input_cols - stride_cols + filter_cols;
                let (mut padding_top, mut padding_bottom, mut padding_left, mut padding_right) =
                    (0, 0, 0, 0);
                match padding_cols % 2 {
                    0 => padding_cols = padding_cols / 2,
                    _ => padding_cols = padding_cols / 2 + 1,
                }
                match padding_cols % 2 {
                    0 => {
                        padding_left = padding_cols / 2;
                        padding_right = padding_cols - padding_left;
                    }
                    _ => {
                        padding_left = padding_cols / 2 + 1;
                        padding_right = padding_cols - padding_left;
                    }
                }

                padded_input =
                    input.pad(&[(padding_top, padding_bottom), (padding_left, padding_right)])?;
            }
            Padding::Custom((top, bottom), (left, right)) => {
                output_rows = (input_rows - filter_rows + top + bottom) / stride_rows + 1;
                output_cols = (input_cols - filter_cols + left + right) / stride_cols + 1;

                padded_input = input.pad(&[(top, bottom), (left, right)])?;
            }
        }
        let mut output = Tensor::zeros(&[output_rows, output_cols]);

        for r in 0..output_rows {
            for c in 0..output_cols {
                let (r_start, r_end) = (r * stride_rows, r * stride_rows + filter_rows - 1);
                let (c_start, c_end) = (c * stride_cols, c * stride_cols + filter_cols - 1);

                *(output.get_element_mut(&[r, c])?) = (padded_input
                    .slice(&[(r_start, r_end), (c_start, c_end)])?
                    * filter.to_owned())
                .sum();
            }
        }
        Ok(output)
    }

    fn convolution(
        input: &Tensor,
        filter: &Tensor,
        stride: (usize, usize),
        padding: Padding,
    ) -> Result<Tensor, AstraError> {
        if input.ndim != 2 || filter.ndim != 2 {
            return Err(AstraError::UnsupportedDimension);
        }

        let (input_rows, input_cols) = (input.shape[0], input.shape[1]);
        let (filter_rows, filter_cols) = (filter.shape[0], filter.shape[1]);
        let (stride_rows, stride_cols) = stride;

        let mut padded_input = input.to_owned();

        let mut output_rows = (input_rows - filter_rows) / stride_rows + 1;
        let mut output_cols = (input_cols - filter_cols) / stride_cols + 1;

        match padding {
            Padding::Valid => {}
            Padding::Same => {
                (output_rows, output_cols) = (input_rows, input_cols);

                let mut padding_rows = (stride_rows - 1) * input_rows - stride_rows + filter_rows;
                match padding_rows % 2 {
                    0 => padding_rows = padding_rows / 2,
                    _ => padding_rows = padding_rows / 2 + 1,
                }
                match padding_rows % 2 {
                    0 => {
                        let padding_top = padding_rows / 2;
                        let padding_bottom = padding_rows - padding_top;
                    }
                    _ => {
                        let padding_top = padding_rows / 2 + 1;
                        let padding_bottom = padding_rows - padding_top;
                    }
                }
                let mut padding_cols = (stride_cols - 1) * input_cols - stride_cols + filter_cols;
                let (mut padding_top, mut padding_bottom, mut padding_left, mut padding_right) =
                    (0, 0, 0, 0);
                match padding_cols % 2 {
                    0 => padding_cols = padding_cols / 2,
                    _ => padding_cols = padding_cols / 2 + 1,
                }
                match padding_cols % 2 {
                    0 => {
                        padding_left = padding_cols / 2;
                        padding_right = padding_cols - padding_left;
                    }
                    _ => {
                        padding_left = padding_cols / 2 + 1;
                        padding_right = padding_cols - padding_left;
                    }
                }

                padded_input =
                    input.pad(&[(padding_top, padding_bottom), (padding_left, padding_right)])?;
            }
            Padding::Custom((top, bottom), (left, right)) => {
                output_rows = (input_rows - filter_rows + top + bottom) / stride_rows + 1;
                output_cols = (input_cols - filter_cols + left + right) / stride_cols + 1;

                padded_input = input.pad(&[(top, bottom), (left, right)])?;
            }
        }
        let mut output = Tensor::zeros(&[output_rows, output_cols]);

        let rotated_filter = filter.to_owned().rotate_180_degrees()?;

        for r in 0..output_rows {
            for c in 0..output_cols {
                let (r_start, r_end) = (r * stride_rows, r * stride_rows + filter_rows - 1);
                let (c_start, c_end) = (c * stride_cols, c * stride_cols + filter_cols - 1);

                *(output.get_element_mut(&[r, c])?) = (padded_input
                    .slice(&[(r_start, r_end), (c_start, c_end)])?
                    * rotated_filter.clone())
                .sum();
            }
        }
        Ok(output)
    }
}

impl Layer for LayerConv2D {
    fn feed_forward(&mut self, inputs: &Tensor) -> Result<Tensor, AstraError> {
        self.input = Some(inputs.to_owned());

        let mut output: Vec<Tensor> = Vec::new();

        let (fs1, fs2) = (self.filter_shape[0], self.filter_shape[1]);

        for f_n in 0..self.filters.shape[0] {
            let mut cur_filter_result: Vec<Tensor> = Vec::with_capacity(self.filters.shape[1]);

            for c_n in 0..self.filters.shape[1] {
                let filter = self
                    .filters
                    .slice(&[(f_n, f_n), (c_n, c_n), (0, fs1 - 1), (0, fs2 - 1)])?
                    .reshape(&[fs1, fs2])?;

                let c_input = inputs
                    .slice(&[
                        (c_n, c_n),
                        (0, inputs.shape[1] - 1),
                        (0, inputs.shape[2] - 1),
                    ])?
                    .reshape(&[inputs.shape[1], inputs.shape[2]])?;

                cur_filter_result.push(Self::convolution(
                    &c_input,
                    &filter,
                    self.stride,
                    self.padding,
                )?);
            }
            let mut summed = Tensor::zeros(&cur_filter_result[0].shape);

            for t in cur_filter_result.into_iter() {
                summed = summed + t;
            }
            output.push(summed);
        }

        self.z = Some(Tensor::stack(&output)?);
        self.output = Some(self.activation.call(self.z.clone().unwrap()));

        Ok(self.output.clone().unwrap())
    }

    fn back_propagation(
        &mut self,
        output_gradient: Tensor,
        learning_rate: f64,
        clipping_value: Option<f64>,
    ) -> Result<Tensor, AstraError> {
        let mut filter_gradients: Vec<Tensor> = Vec::new();
        let mut input_gradients: Vec<Tensor> = Vec::new();

        let layer_output = self
            .output
            .clone()
            .ok_or(AstraError::UninitializedLayerParameter(
                "self.output".to_string(),
            ))?;

        let layer_input = self
            .input
            .to_owned()
            .ok_or(AstraError::UninitializedLayerParameter(
                "self.input".to_string(),
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

            let cur_filter_output_gradient = output_gradient.slice(&[
                (f_n, f_n),
                (0, output_gradient.shape[1] - 1),
                (0, output_gradient.shape[2] - 1),
            ])?;

            let layer_output_cur_filter = layer_output.slice(&[
                (f_n, f_n),
                (0, layer_output.shape[1] - 1),
                (0, layer_output.shape[2] - 1),
            ])?
            .reshape(&[layer_output.shape[1], layer_output.shape[2]])?;

            let mut input_gradient = Tensor::zeros(&cur_filter_output_gradient.shape);

            for c_n in 0..self.filter_shape[1] {
                let input_channel = layer_input
                    .slice(&[
                        (c_n, c_n),
                        (0, layer_input.shape[1] - 1),
                        (0, layer_input.shape[2] - 1),
                    ])?
                    .reshape(&[layer_input.shape[1], layer_input.shape[2]])?;

                let gradient = Self::cross_correlation(
                    &input_channel,
                    &layer_output_cur_filter,
                    self.stride,
                    Padding::Valid,
                )?;
                filter_gradients.push(gradient.clone());

                let filter_channel = cur_filter
                    .slice(&[
                        (c_n, c_n),
                        (0, cur_filter.shape[1] - 1),
                        (0, cur_filter.shape[2] - 1),
                    ])?
                    .reshape(&[cur_filter.shape[1], cur_filter.shape[2]])?;

                input_gradient = input_gradient
                    + Self::convolution(
                        &cur_filter_output_gradient,
                        &filter_channel,
                        self.stride,
                        Padding::Same,
                    )?;
            }
            input_gradients.push(input_gradient);
        }

        // Update filter weights
        let filter_gradients_stacked = match clipping_value {
            None => Tensor::stack(&filter_gradients)?,
            Some(v) => Tensor::stack(&filter_gradients)?.clip(v),
        };
        self.filters = self.filters.to_owned() - (filter_gradients_stacked * learning_rate);

        // Sum errors for the previous layer
        let input_gradients = input_gradients
            .into_iter()
            .reduce(|a, b| a + b)
            .ok_or(AstraError::CustomError("could not sum errors".to_string()))?;

        Ok(input_gradients)
    }
}
