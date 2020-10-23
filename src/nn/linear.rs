use crate::tensor::Tensor;
#[derive(Debug, Clone, Copy)]
pub struct LinearConfig {
    pub ws_init: super::Init,
    pub bs_init: Option<super::Init>,
    pub bias: bool,
    pub in_features: usize,
    pub out_features: usize,
}

impl Default for LinearConfig {
    fn default() -> Self {
        LinearConfig {
            ws_init: super::Init::KaimingUniform,
            bs_init: None,
            bias: true,
            in_features: 0,
            out_features: 0,
        }
    }
}
#[derive(Debug)]
pub struct Linear {
    pub ws: Option<Tensor>,
    pub bs: Option<Tensor>,
    options: LinearConfig,
}

impl Linear {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let options = LinearConfig {
            in_features: in_dim,
            out_features: out_dim,
            ..LinearConfig::default()
        };
        let mut self_ = Self {
            ws: None,
            bs: None,
            options,
        };
        self_.reset();
        self_
    }

    fn reset(&mut self) {
        let ws = Tensor::empty(&[self.options.out_features, self.options.in_features]);
        super::module::register_parameter(&ws, true);
        self.ws = Some(ws);
        if self.options.bias {
            let bs_ = Tensor::empty(&[self.options.out_features]);
            super::module::register_parameter(&bs_, true);
            self.bs = Some(bs_);
        };
        self.reset_parameters();
    }
    fn reset_parameters(&mut self) {
        crate::nn::init::kaiming_uniform_(
            self.ws.as_ref().unwrap(),
            (5.0f64).sqrt(),
            crate::nn::init::FanModeType::FanIn,
            crate::nn::init::NonlinerityType::LeakyReLU,
        );
        if let Some(bs) = self.bs.as_ref() {
            let (fan_in, _fan_out) =
                crate::nn::init::calculate_fan_in_fan_out(self.ws.as_ref().unwrap());
            let bound = 1.0 / (fan_in as f64).sqrt();
            bs.uniform_(-bound, bound)
        }
    }
}

impl super::module::Module for Linear {
    fn forward(&self, xs: &[&Tensor]) -> Tensor {
        let result = xs[0].matmul(&self.ws.as_ref().unwrap().t(), true);
        if let Some(bs) = self.bs.as_ref() {
            result + bs.clone()
        } else {
            result
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        if self.bs.is_some() {
            vec![
                self.ws.as_ref().unwrap().clone(),
                self.bs.as_ref().unwrap().clone(),
            ]
        } else {
            vec![self.ws.as_ref().unwrap().clone()]
        }
    }
}

#[cfg(test)]
mod test {
    use super::Linear;
    use crate::autograd::backward;
    use crate::nn::Module;
    use crate::tensor::Tensor;

    #[test]
    fn linear_backward_test() {
        let linear = Linear::new(4, 3);
        let x = Tensor::from_scalar(&[2, 4], 1.5, true);
        let y = linear.forward(&[&x]);
        println!("Result: {:?}", y);
        backward::backward(&vec![y], &vec![], false);

        println!("Input Grad: {:?}", x.grad());

        // let ws_grad = linear.ws.get_tensor_impl().grad();
        // assert!(ws_grad.is_some());
        // assert!(
        //     ws_grad
        //         .as_ref()
        //         .unwrap()
        //         .borrow()
        //         .get_tensor_impl()
        //         .data
        //         .ndim()
        //         == 2
        // );
        // let result = ws_grad
        //     .unwrap()
        //     .borrow()
        //     .get_tensor_impl()
        //     .data
        //     .as_slice()
        //     .unwrap()
        //     .to_vec();
        // let expected = vec![4.0, 4.0];
        // assert_eq!(result, expected);
    }
}
