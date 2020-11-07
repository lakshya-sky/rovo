use crate::tensor::{NewTensor, NewTensorIterator};
pub fn add_out<'a>(result: &'a NewTensor, self_: &NewTensor, other: &NewTensor) -> &'a NewTensor {
    let mut iter = NewTensorIterator::binary_op(result, self_, other, true);
    super::cpu::binary_ops_kernel::add_kernel(&mut iter);
    return result;
}
pub fn add(self_: &NewTensor, other: &NewTensor, _alpha: f32) -> NewTensor {
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
pub fn sub(self_: &NewTensor, other: &NewTensor, _alpha: f32) -> NewTensor {
    let result = NewTensor::default();
    let mut iter = NewTensorIterator::binary_op(&result, self_, other, false);
    super::cpu::binary_ops_kernel::sub_kernel(&mut iter);
    return result;
}
