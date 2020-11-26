use crate::c10::Scalar;
use crate::tensor::{Tensor, TensorIterator};

pub fn fill_(self_: &Tensor, value: impl Into<Scalar>) -> &Tensor {
    let iter = TensorIterator::nullary_op(self_);
    super::cpu::fill_kernel(iter, value.into());
    self_
}
