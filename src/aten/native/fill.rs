use crate::c10::Scalar;
use crate::tensor::{NewTensor, NewTensorIterator};

pub fn fill_(self_: &NewTensor, value: impl Into<Scalar>) -> &NewTensor {
    let iter = NewTensorIterator::nullary_op(self_);
    super::cpu::fill_kernel::fill_kernel(iter, value.into());
    self_
}
