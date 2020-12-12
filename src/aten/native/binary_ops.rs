use crate::c10::Scalar;
use crate::tensor::{Tensor, TensorIterator};
use crate::{aten::scalar_to_tensor, c10::DeviceType};

use super::{
    cpu::{add_kernel, sigmoid_backward_kernel},
    DispatchStub,
};

#[inline(always)]
fn binary_op_impl_out<'a>(
    result: &'a Tensor,
    self_: &Tensor,
    other: &Tensor,
    stub: impl DispatchStub,
) -> &'a Tensor {
    let iter = TensorIterator::binary_op(result, self_, other, false);
    stub.call(iter.device_type(0), iter);
    result
}

#[inline(always)]
fn binary_op_impl<'b, F>(self_: &'b Tensor, other: &Tensor, out_impl: F) -> Tensor
where
    F: for<'a> Fn(&'a Tensor, &Tensor, &Tensor) -> &'a Tensor,
{
    let result = Tensor::default();
    out_impl(&result, self_, other);
    result
}

// #[inline(always)]
// fn unary_op_impl_<'a: 'b, 'b, F>(self_: &'a Tensor, out_impl: F) -> &'a Tensor
// where
//     F: Fn(&'a Tensor, &'b Tensor) -> &'a Tensor,
// {
//     out_impl(self_, self_)
// }

/* ---------------------------- Stubs --------------------------*/
struct add_stub;

/* -------------------------- Stubs End ------------------------*/

pub fn add_out<'a>(
    result: &'a Tensor,
    self_: &Tensor,
    other: &Tensor,
    alpha: Scalar,
) -> &'a Tensor {
    let mut iter = TensorIterator::binary_op(result, self_, other, true);
    add_kernel(&mut iter, alpha);
    return result;
}

pub fn add(self_: &Tensor, other: &Tensor, alpha: impl Into<Scalar>) -> Tensor {
    let alpha: Scalar = alpha.into();
    let result = Tensor::default();
    let mut iter = TensorIterator::binary_op(&result, self_, other, false);
    add_kernel(&mut iter, alpha);
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

pub fn sigmoid_backward_out<'r>(
    result: &'r Tensor,
    grad_output: &Tensor,
    output: &Tensor,
) -> &'r Tensor {
    // let mut iter = TensorIterator::binary_op(result, grad_output, output, false);
    // sigmoid_backward_stub(iter.device_type(0), &mut iter);
    // return result;
    binary_op_impl_out(result, grad_output, output, sigmoid_backward_stub)
}

pub fn sigmoid_backward(grad_output: &Tensor, output: &Tensor) -> Tensor {
    binary_op_impl(grad_output, output, sigmoid_backward_out)
    // let mut iter = TensorIterator::binary_op(&result, grad_output, output, false);
    // sigmoid_backward_stub(iter.device_type(0), &mut iter);
    // return result;
}

/* --------------------------- Scalar ------------------------------- */
pub fn add_scalar(self_: &Tensor, other: impl Into<Scalar>, alpha: impl Into<Scalar>) -> Tensor {
    add(self_, &wrapped_scalar_tensor(other.into()), alpha.into())
}

pub fn div_scalar(self_: &Tensor, other: impl Into<Scalar>) -> Tensor {
    div(self_, &wrapped_scalar_tensor(other.into()))
}

pub fn mul_scalar(self_: &Tensor, other: impl Into<Scalar>) -> Tensor {
    mul(self_, &wrapped_scalar_tensor(other.into()))
}

pub fn sub_scalar(self_: &Tensor, other: impl Into<Scalar>, alpha: impl Into<Scalar>) -> Tensor {
    sub(self_, &wrapped_scalar_tensor(other.into()), alpha.into())
}

#[inline(always)]
pub fn wrapped_scalar_tensor(scalar: Scalar) -> Tensor {
    let tensor = scalar_to_tensor(scalar, None);
    tensor.get_unsafe_tensor_impl().set_wrapped_number(true);
    tensor
}

struct sigmoid_backward_stub;

impl DispatchStub for sigmoid_backward_stub {
    fn call(&self, _device_type: DeviceType, mut iter: TensorIterator) {
        sigmoid_backward_kernel(&mut iter)
    }
}
