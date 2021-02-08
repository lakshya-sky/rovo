use crate::{c10::ScalarType, tensor::Tensor};

mod detail {
    use super::*;
    #[inline(always)]
    pub fn log_softmax(input: &Tensor, dim: i64, dtype: Option<ScalarType>) -> Tensor {
        input.log_softmax(dim, dtype)
    }
}
pub struct LogSoftmaxFuncOptions {
    dim: i64,
    dtype: Option<ScalarType>,
}

impl LogSoftmaxFuncOptions {
    pub fn new(dim: i64) -> Self {
        Self { dim, dtype: None }
    }
}

impl From<i64> for LogSoftmaxFuncOptions {
    fn from(d: i64) -> Self {
        Self::new(d)
    }
}

// Computes Log Softmax of the tensor. passing an i64 will convert
// it to LogSoftmaxFuncOptions with dtype=None.
pub fn log_softmax(input: &Tensor, options: impl Into<LogSoftmaxFuncOptions>) -> Tensor {
    let options = options.into();
    detail::log_softmax(input, options.dim, options.dtype)
}
