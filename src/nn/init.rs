//! Variable initialization.
use crate::core::*;
use crate::tensor::Tensor;
#[derive(Debug, Copy, Clone)]
pub enum Init {
    /// Constant value.
    Const(f64),

    /// Random normal with some mean and standard deviation.
    Randn { mean: f64, stdev: f64 },

    /// Uniform initialization between some lower and upper bounds.
    Uniform { lo: f64, up: f64 },

    /// Kaiming uniform initialization.
    KaimingUniform,
}

pub enum FanModeType {
    FanIn,
    FanOut,
}
pub enum NonlinerityType {
    Linear,
    Conv1D,
    Conv2D,
    Conv3D,
    ConvTranspose1D,
    ConvTranspose2D,
    ConvTranspose3D,
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU,
}

pub struct Fan {
    in_: usize,
    out_: usize,
}

impl Fan {
    pub fn new(tensor: &Tensor) -> Self {
        let in_;
        let out_;
        // Below commented code is actual implementation, needs be added after [] oprator is
        // implemented for Tensor. For now the function supports only tensor with two dimensions.
        // let dimensions = tensor.dim();
        // if dimensions == 2 {
        //    in_ = tensor.size(1);
        //    out_ = tensor.size(0);
        // } else {
        //    in_ = tensor.size(1) * tensor[0][0].numel();
        //    out_ = tensor.size(0) * tensor[0][0].numel();
        // }
        in_ = tensor.size(1);
        out_ = tensor.size(0);
        Self { in_, out_ }
    }
}

pub fn calculate_fan_in_fan_out(tensor: &Tensor) -> (usize, usize) {
    let in_ = tensor.size(1);
    let out_ = tensor.size(0);
    (in_, out_)
}

pub fn kaiming_uniform_<'a>(
    tensor: &'a Tensor,
    a: f64,
    mode: FanModeType,
    non_linearity: NonlinerityType,
) -> &'a Tensor {
    let _ = NoGradGuard::default();

    let std_ = calculate_kaiming_std(&tensor, a, mode, non_linearity);
    let bound = (3.0f64).sqrt() * std_;
    tensor.uniform(-bound, bound);
    tensor
}

pub fn calculate_kaiming_std(
    tensor: &Tensor,
    a: f64,
    mode: FanModeType,
    non_linearity: NonlinerityType,
) -> f64 {
    let _ = NoGradGuard::default();
    let fan = Fan::new(tensor);
    let gain = calculate_gain(non_linearity, a);
    let std_;
    std_ = match mode {
        FanModeType::FanIn => (gain / (fan.in_ as f64).sqrt()),
        FanModeType::FanOut => (gain / (fan.out_ as f64).sqrt()),
    };
    std_
}

fn calculate_gain(non_linearity: NonlinerityType, param: f64) -> f64 {
    match non_linearity {
        NonlinerityType::Tanh => 5.0 / 3.0,
        NonlinerityType::ReLU => (2.0f64).sqrt(),
        NonlinerityType::LeakyReLU => (2.0 / (1.0 + param.powi(2))).sqrt(),
        _ => 1.0,
    }
}

// #[cfg(test)]
// mod test {
//     use super::*;
//     use crate::core::manual_seed;
//     #[test]
//     fn test_uniform() {
//         manual_seed(0);
//         let tnsr = Tensor::empty(&[1, 2]);
//         tnsr.uniform_(0.0, 1.0);
//         println!("{:?}", tnsr);
//     }
//     #[test]
//     fn test_kaiming_uniform() {
//         manual_seed(0);
//         let tnsr = Tensor::empty(&[3, 4]);
//         let _ = kaiming_uniform_(
//             &tnsr,
//             (5.0f64).sqrt(),
//             FanModeType::FanIn,
//             NonlinerityType::LeakyReLU,
//         );
//         println!("{:?}", tnsr);
//     }
// }
