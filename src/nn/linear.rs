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
    fn forward(&self, xs: &Tensor) -> Tensor {
        &xs.matmul(&self.ws.t(), true) + &self.bs
    }
}

#[cfg(test)]
mod test {
    use super::{Linear, LinearConfig};
    use crate::autograd::backward;
    use crate::nn::Module;
    use crate::tensor::Tensor;

    #[test]
    fn it_works() {
        let config = LinearConfig::default();
        let linear = Linear::new(2, 1, config);
        let x = Tensor::from_scalar(&[2, 2], 2.0, true);
        let y = linear.forward(&x);
        backward::backward(&vec![y], &vec![], false);
        println!("Input Teensor Grad {:?}", x.get_tensor_impl().grad());
        println!("Linear Tensor Grad {:?}", linear.ws.get_tensor_impl().grad());
    }
}
