use crate::aten::scalar_to_tensor;
use crate::c10::Scalar;
use crate::tensor::{NewTensor, NewTensorIterator};

pub fn add_out<'a>(result: &'a NewTensor, self_: &NewTensor, other: &NewTensor) -> &'a NewTensor {
    let mut iter = NewTensorIterator::binary_op(result, self_, other, true);
    super::cpu::binary_ops_kernel::add_kernel(&mut iter);
    return result;
}
pub fn add(self_: &NewTensor, other: &NewTensor, _alpha: impl Into<Scalar>) -> NewTensor {
    let result = NewTensor::default();
    let mut iter = NewTensorIterator::binary_op(&result, self_, other, false);
    super::cpu::binary_ops_kernel::add_kernel(&mut iter);
    return result;
}
pub fn div_out<'a>(result: &'a NewTensor, self_: &NewTensor, other: &NewTensor) -> &'a NewTensor {
    let mut iter = NewTensorIterator::binary_op(result, self_, other, true);
    super::cpu::binary_ops_kernel::div_kernel(&mut iter);
    return result;
}
pub fn div(self_: &NewTensor, other: &NewTensor) -> NewTensor {
    let result = NewTensor::default();
    let mut iter = NewTensorIterator::binary_op(&result, self_, other, false);
    super::cpu::binary_ops_kernel::div_kernel(&mut iter);
    return result;
}
pub fn mul_out<'a>(result: &'a NewTensor, self_: &NewTensor, other: &NewTensor) -> &'a NewTensor {
    let mut iter = NewTensorIterator::binary_op(result, self_, other, true);
    super::cpu::binary_ops_kernel::mul_kernel(&mut iter);
    return result;
}
pub fn mul(self_: &NewTensor, other: &NewTensor) -> NewTensor {
    let result = NewTensor::default();
    let mut iter = NewTensorIterator::binary_op(&result, self_, other, false);
    super::cpu::binary_ops_kernel::mul_kernel(&mut iter);
    return result;
}
pub fn sub_out<'a>(result: &'a NewTensor, self_: &NewTensor, other: &NewTensor) -> &'a NewTensor {
    let mut iter = NewTensorIterator::binary_op(result, self_, other, true);
    super::cpu::binary_ops_kernel::sub_kernel(&mut iter);
    return result;
}
pub fn sub(self_: &NewTensor, other: &NewTensor, _alpha: impl Into<Scalar>) -> NewTensor {
    let result = NewTensor::default();
    let mut iter = NewTensorIterator::binary_op(&result, self_, other, false);
    super::cpu::binary_ops_kernel::sub_kernel(&mut iter);
    return result;
}

/* --------------------------- Scalar ------------------------------- */
pub fn add_scalar(
    self_: &NewTensor,
    other: impl Into<Scalar>,
    alpha: impl Into<Scalar>,
) -> NewTensor {
    add(self_, &wrapped_scalar_tensor(other.into()), alpha.into())
}

pub fn mul_scalar(self_: &NewTensor, other: impl Into<Scalar>) -> NewTensor {
    mul(self_, &wrapped_scalar_tensor(other.into()))
}

pub fn sub_scalar(
    self_: &NewTensor,
    other: impl Into<Scalar>,
    alpha: impl Into<Scalar>,
) -> NewTensor {
    sub(self_, &wrapped_scalar_tensor(other.into()), alpha.into())
}

#[inline(always)]
fn wrapped_scalar_tensor(scalar: Scalar) -> NewTensor {
    let tensor = scalar_to_tensor(scalar, None);
    tensor.get_unsafe_tensor_impl().set_wrapped_number(true);
    tensor
}
