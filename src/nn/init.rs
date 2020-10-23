//! Variable initialization.
use crate::core::*;
use crate::tensor::Tensor;

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
/// Variable initializations.
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

pub fn init(i: Init, dims: &[usize]) -> Tensor {
    match i {
        Init::Const(cst) => {
            if cst == 0. {
                Tensor::zeros(dims)
            } else if (cst - 1.).abs() <= std::f64::EPSILON {
                Tensor::ones(dims)
            } else {
                Tensor::ones(dims) * cst
            }
        }

        Init::Uniform { lo, up } => Tensor::uniform(dims, lo, up),

        Init::Randn { mean, stdev } => {
            if mean == 0. && (stdev - 1.).abs() <= std::f64::EPSILON {
                Tensor::randn(dims)
            } else {
                Tensor::randn(dims) * stdev + mean
            }
        }

        Init::KaimingUniform => {
            let fan_in: usize = dims.iter().skip(1).product();
            let bound = (1.0 / fan_in as f64).sqrt();
            Tensor::uniform(dims, -bound, bound)
        }
    }
}

// impl Init {
//     /// Re-initializes an existing tensor with the specified initialization
//     pub fn set(self, tensor: &mut Tensor) {
//         match self {
//             Init::Const(cst) => {
//                 let _ = tensor.fill_(cst);
//             }
//             Init::Uniform { lo, up } => {
//                 let _ = tensor.uniform_(lo, up);
//             }
//             Init::KaimingUniform => {
//                 let fan_in: i64 = tensor.size().iter().skip(1).product();
//                 let bound = (1.0 / fan_in as f64).sqrt();
//                 let _ = tensor.uniform_(-bound, bound);
//             }
//             Init::Randn { mean, stdev } => {
//                 tensor.copy_(&(tensor.randn_like() * stdev + mean));
//             }
//         }
//     }
// }

// impl Tensor {
//     /// Re-initializes the tensor using the specified initialization.
//     pub fn init(&mut self, i: Init) {
//         i.set(self)
//     }
// }

pub fn calculate_fan_in_fan_out(tensor: &Tensor) -> (usize, usize) {
    let in_ = tensor.size(1);
    let out_ = tensor.size(0);
    (in_, out_)
}

pub fn kaiming_uniform_(
    tensor: &Tensor,
    a: f64,
    mode: FanModeType,
    non_linearity: NonlinerityType,
) -> Tensor {
    let _ = NoGradGuard::default();

    let std_ = calculate_kaiming_std(&tensor, a, mode, non_linearity);
    let bound = (3.0f64).sqrt() * std_;
    tensor.uniform_(-bound, bound);
    tensor.clone()
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::core::manual_seed;
    #[test]
    fn test_uniform() {
        manual_seed(0);
        let tnsr = Tensor::empty(&[1, 2]);
        tnsr.uniform_(0.0, 1.0);
        println!("{:?}", tnsr);
    }
    #[test]
    fn test_kaiming_uniform() {
        manual_seed(0);
        let tnsr = Tensor::empty(&[3, 4]);
        let _ = kaiming_uniform_(
            &tnsr,
            (5.0f64).sqrt(),
            FanModeType::FanIn,
            NonlinerityType::LeakyReLU,
        );
        println!("{:?}", tnsr);
    }
}
