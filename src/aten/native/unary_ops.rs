use crate::aten::native;
use crate::tensor::{Tensor, TensorIterator};

// struct neg_stub;

// impl DispatchStub for neg_stub {}

// #[inline(always)]
// fn unary_op_impl_out<'a, Stub>(
//     result: &'a Tensor,
//     self_: &Tensor,
//     stub: impl DispatchStub,
// ) -> &'a Tensor where {
//     let iter = TensorIterator::unary_op(result, self_, true);
//     stub.call(iter.device_type(), iter);
//     result
// }

#[inline(always)]
fn unary_op_impl<'a, 'b, F>(self_: &'b Tensor, _out_impl: F) -> Tensor
where
    F: Fn(&'a Tensor, &'b Tensor) -> &'a Tensor + 'a,
{
    let result = native::empty(&[0], self_.options(), None);
    // out_impl(&result, self_);

    // Todo: Replace this with above code after you find the solution.
    neg_out(&result, self_);

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
    // unary_op_impl_out(result, self_, neg_stub)
    let mut iter = TensorIterator::unary_op(result, self_, true);
    native::cpu::neg_kernel(&mut iter);
    result
}

pub fn neg(self_: &Tensor) -> Tensor {
    unary_op_impl(self_, neg_out)
}

pub fn neg_<'a>(self_: &'a Tensor) -> &'a Tensor {
    unary_op_impl_(self_, neg_out)
}
