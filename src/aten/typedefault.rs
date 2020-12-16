use crate::{c10::ScalarType, tensor::Tensor};

use super::native;

pub fn log_softmax_int(self_: &Tensor, dim: i64, dtype: Option<ScalarType>) -> Tensor {
    native::log_softmax(self_, dim, dtype)
}
