use crate::tensor::{NewTensor, NewTensorIterator};
pub fn fill_(self_: &NewTensor, value: f32) -> &NewTensor {
    let iter = NewTensorIterator::nullary_op(self_);
    super::cpu::fill_kernel::fill_kernel(iter, value);
    self_
}
