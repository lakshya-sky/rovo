use crate::aten::native::cpu;
use crate::core::Generator;
use crate::tensor::{NewTensor, NewTensorIterator};

pub fn uniform_impl_(self_: &NewTensor, from: f64, to: f64, generator: Option<Generator>) {
    // Todo: Pytorch does boundary checking on the from and to values here.

    let iter = NewTensorIterator::nullary_op(self_);
    let gen_ = match generator {
        Some(g) => Some(g.as_with_cpu_impl()),
        None => None,
    };
    cpu::unary_ops_kernel::uniform_kernel(iter, from, to, gen_);
}
