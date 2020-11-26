use crate::aten::scalar_to_tensor;
use crate::c10::Scalar;
use crate::tensor::{Tensor, TensorIterator};

pub fn add_out<'a>(result: &'a Tensor, self_: &Tensor, other: &Tensor) -> &'a Tensor {
    let mut iter = TensorIterator::binary_op(result, self_, other, true);
    super::cpu::add_kernel(&mut iter);
    return result;
}
pub fn add(self_: &Tensor, other: &Tensor, _alpha: impl Into<Scalar>) -> Tensor {
    let result = Tensor::default();
    let mut iter = TensorIterator::binary_op(&result, self_, other, false);
    super::cpu::add_kernel(&mut iter);
    return result;
}
pub fn div_out<'a>(result: &'a Tensor, self_: &Tensor, other: &Tensor) -> &'a Tensor {
    let mut iter = TensorIterator::binary_op(result, self_, other, true);
    super::cpu::div_kernel(&mut iter);
    return result;
}
pub fn div(self_: &Tensor, other: &Tensor) -> Tensor {
    let result = Tensor::default();
    let mut iter = TensorIterator::binary_op(&result, self_, other, false);
    super::cpu::div_kernel(&mut iter);
    return result;
}
pub fn mul_out<'a>(result: &'a Tensor, self_: &Tensor, other: &Tensor) -> &'a Tensor {
    let mut iter = TensorIterator::binary_op(result, self_, other, true);
    super::cpu::mul_kernel(&mut iter);
    return result;
}
pub fn mul(self_: &Tensor, other: &Tensor) -> Tensor {
    let result = Tensor::default();
    let mut iter = TensorIterator::binary_op(&result, self_, other, false);
    super::cpu::mul_kernel(&mut iter);
    return result;
}
pub fn sub_out<'a>(result: &'a Tensor, self_: &Tensor, other: &Tensor) -> &'a Tensor {
    let mut iter = TensorIterator::binary_op(result, self_, other, true);
    super::cpu::sub_kernel(&mut iter);
    return result;
}
pub fn sub(self_: &Tensor, other: &Tensor, _alpha: impl Into<Scalar>) -> Tensor {
    let result = Tensor::default();
    let mut iter = TensorIterator::binary_op(&result, self_, other, false);
    super::cpu::sub_kernel(&mut iter);
    return result;
}

/* --------------------------- Scalar ------------------------------- */
pub fn add_scalar(
    self_: &Tensor,
    other: impl Into<Scalar>,
    alpha: impl Into<Scalar>,
) -> Tensor {
    add(self_, &wrapped_scalar_tensor(other.into()), alpha.into())
}

pub fn mul_scalar(self_: &Tensor, other: impl Into<Scalar>) -> Tensor {
    mul(self_, &wrapped_scalar_tensor(other.into()))
}

pub fn sub_scalar(
    self_: &Tensor,
    other: impl Into<Scalar>,
    alpha: impl Into<Scalar>,
) -> Tensor {
    sub(self_, &wrapped_scalar_tensor(other.into()), alpha.into())
}

#[inline(always)]
fn wrapped_scalar_tensor(scalar: Scalar) -> Tensor {
    let tensor = scalar_to_tensor(scalar, None);
    tensor.get_unsafe_tensor_impl().set_wrapped_number(true);
    tensor
}
