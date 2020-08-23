//! Variable initialization.
use crate::tensor::Tensor;

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
