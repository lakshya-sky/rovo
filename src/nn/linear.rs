use crate::tensor::Tensor;
#[derive(Debug, Clone, Copy)]
pub struct LinearConfig {
    pub ws_init: super::Init,
    pub bs_init: Option<super::Init>,
    pub bias: bool,
}

impl Default for LinearConfig {
    fn default() -> Self {
        LinearConfig {
            ws_init: super::Init::KaimingUniform,
            bs_init: None,
            bias: true,
        }
    }
}
#[derive(Debug)]
pub struct Linear {
    pub ws: Tensor,
    pub bs: Tensor,
}

impl Linear {
    pub fn new(in_dim: usize, out_dim: usize, c: LinearConfig) -> Self {
        let bs = if c.bias {
            let bs_init = c.bs_init.unwrap_or_else(|| {
                let bound = 1.0 / (in_dim as f64).sqrt();
                super::Init::Uniform {
                    lo: -bound,
                    up: bound,
                }
            });
            super::init::init(bs_init, &[out_dim])
        } else {
            Tensor::zeros(&[out_dim])
        };

        let ws = super::init::init(c.ws_init, &[out_dim, in_dim]);
        super::module::register_parameter(&ws, true);
        super::module::register_parameter(&bs, true);
        Linear { ws: ws, bs }
    }
}

impl super::module::Module for Linear {
    fn forward(&self, xs: &[&Tensor]) -> Tensor {
        &xs[0].matmul(&self.ws.t(), true) + &self.bs
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.ws.clone(), self.bs.clone()]
    }
}

#[cfg(test)]
mod test {
    use super::{Linear, LinearConfig};
    use crate::autograd::backward;
    use crate::nn::Module;
    use crate::tensor::Tensor;

    #[test]
    fn linear_backward_test() {
        let config = LinearConfig::default();
        let linear = Linear::new(4, 3, config);
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
