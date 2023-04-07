use crate::astra_net::activation::Activation;
use crate::astra_net::layer::Layer;
use crate::tensor::Tensor;

use ndarray_rand::rand_distr::{Distribution, Normal};

pub struct LayerConv2D {
    pub filters: Vec<Tensor>,
    pub kernal_shape: Vec<usize>,
    pub stride: usize,
    pub padding: usize,
    pub input_shape: Vec<usize>,
    pub activation: Box<dyn Activation>,
    pub input: Option<Tensor>,
    pub output: Option<Tensor>,
}

impl LayerConv2D {
    fn new() -> Self {
        todo!()
    }

    fn convolution(&self, input: &Tensor, kernel: &Tensor) -> Tensor {
        assert!(input.ndim == 2, "input should be a matrix");
        assert!(kernel.ndim == 2, "kernel should be a matrix");

        let (img_height, img_width) = (input.shape[0], input.shape[1]);
        let (kernel_height, kernel_width) = (kernel.shape[0], kernel.shape[1]);

        let output_height = (img_height - kernel_height + 2 * self.padding) / (self.stride + 1);
        let output_width = (img_width - kernel_width + 2 * self.padding) / (self.stride + 1);

        let mut output = Tensor::zeros(&[output_height, output_width]);

        let mut padded_image = input.to_owned();

        match self.padding {
            0 => {}
            _ => {
                padded_image = padded_image
                    .pad(&[(self.padding, self.padding), (self.padding, self.padding)])
                    .unwrap()
            }
        }

        for y in 0..output_height {
            for x in 0..output_width {
                let (y_start, y_end) = (y * self.stride, y * self.stride + kernel_height);
                let (x_start, x_end) = (x * self.stride, x * self.stride + kernel_width);
                *(output.get_element_mut(&[y, x]).unwrap()) = (padded_image
                    .slice(&[(y_start, y_end), (x_start, x_end)])
                    .unwrap()
                    * kernel.to_owned())
                .sum();
            }
        }
        output
    }
}

impl Layer for LayerConv2D {
    fn feed_forward(&mut self, inputs: &Tensor) -> Tensor {
        let mut output: Vec<Tensor> = Vec::new();

        for filter in self.filters.clone().into_iter() {
            output.push(self.convolution(inputs, &filter));
        }
        Tensor::stack(&output, -1).unwrap()
    }

    fn back_propagation(&mut self, error: Tensor, learning_rate: f64) -> Tensor {
        todo!()
    }
}
