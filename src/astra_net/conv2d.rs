use crate::astra_net::activation::Activation;
use crate::astra_net::layer::Layer;
use crate::astra_net::net_error::NetError;
use crate::tensor::Tensor;

use ndarray_rand::rand_distr::{Distribution, Normal};

pub struct LayerConv2D {
    pub filters: Vec<Tensor>,
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
        n_filters: usize,
        padding: usize,
        stride: usize,
        activation: Box<dyn Activation>,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(-1.0, 1.0).unwrap();

        let filters = (0..n_filters)
            .into_iter()
            .map(|_n| Tensor::from_fn(kernel_shape.clone(), || normal.sample(&mut rng)))
            .collect();

        Self {
            filters,
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
        Ok(output)
    }
}

impl Layer for LayerConv2D {
    fn feed_forward(&mut self, inputs: &Tensor) -> Result<Tensor, NetError> {
        let mut output: Vec<Tensor> = Vec::new();

        for filter in self.filters.clone().into_iter() {
            output.push(self.convolution(inputs, &filter)?);
        }
        Tensor::stack(&output).map_err(NetError::TensorBasedError)
    }

    fn back_propagation(&mut self, error: Tensor, learning_rate: f64) -> Result<Tensor, NetError> {
        todo!()
    }
}
