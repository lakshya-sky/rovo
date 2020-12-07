use num::FromPrimitive;

use crate::tensor::{Tensor, TensorIterator};
use crate::{
    c10::{Scalar, ScalarType},
    AT_DISPATCH_ALL_TYPES_AND, AT_PRIVATE_CASE_TYPE,
};

fn fill_fast<T: FromPrimitive>(self_: &Tensor, value: Scalar) {
    let value = value.to::<T>();
    let dptr = self_.data_ptr().as_ptr() as *mut T;
    unsafe {
        *dptr = value;
    }
}

pub fn fill_out(self_: &Tensor, value: Scalar) {
    //Todo: add complex number condition here
    if self_.device().is_cpu() && self_.numel() == 1 {
        AT_DISPATCH_ALL_TYPES_AND!(self_.scalar_type(), "fill_out", || {
            fill_fast::<SCALART>(self_, value);
        });
    } else {
        let iter = TensorIterator::nullary_op(self_);
        super::cpu::fill_kernel(iter, value.into());
    }
}
pub fn fill_(self_: &Tensor, value: impl Into<Scalar>) {
    fill_out(self_, value.into())
}
