use native::{cpu::*, DispatchStub};

use crate::tensor::{Tensor, TensorIterator};
use crate::{aten::native, c10::DeviceType};

struct neg_stub;
struct sigmoid_stub;

impl DispatchStub for neg_stub {
    fn call(&self, _device_type: DeviceType, mut iter: TensorIterator) {
        neg_kernel(&mut iter);
    }
}

impl DispatchStub for sigmoid_stub {
    fn call(&self, _device_type: DeviceType, mut iter: TensorIterator) {
        sigmoid_kernel(&mut iter);
    }
}

#[inline(always)]
fn unary_op_impl_out<'a>(
    result: &'a Tensor,
    self_: &Tensor,
    stub: impl DispatchStub,
) -> &'a Tensor {
    let iter = TensorIterator::unary_op(result, self_, true);
    stub.call(iter.device_type(0), iter);
    result
}

#[inline(always)]
fn unary_op_impl<'b, F>(self_: &'b Tensor, out_impl: F) -> Tensor
where
    F: for<'a> Fn(&'a Tensor, &'b Tensor) -> &'a Tensor,
{
    let result = native::empty(&[0], self_.options(), None);
    out_impl(&result, self_);
    result
}

#[inline(always)]
fn unary_op_impl_<'a: 'b, 'b, F>(self_: &'a Tensor, out_impl: F) -> &'a Tensor
where
    F: Fn(&'a Tensor, &'b Tensor) -> &'a Tensor,
{
    out_impl(self_, self_)
}

pub fn neg_out<'a, 'b>(result: &'a Tensor, self_: &'b Tensor) -> &'a Tensor {
    unary_op_impl_out(result, self_, neg_stub)
}

pub fn neg(self_: &Tensor) -> Tensor {
    unary_op_impl(self_, neg_out)
}

pub fn neg_<'a>(self_: &'a Tensor) -> &'a Tensor {
    unary_op_impl_(self_, neg_out)
}

pub fn sigmoid_out<'a, 'b>(result: &'a Tensor, self_: &'b Tensor) -> &'a Tensor {
    unary_op_impl_out(result, self_, sigmoid_stub)
}
pub fn sigmoid(self_: &Tensor) -> Tensor {
    unary_op_impl(self_, sigmoid_out)
}
pub fn sigmoid_<'a>(self_: &'a Tensor) -> &'a Tensor {
    unary_op_impl_(self_, sigmoid_out)
}
