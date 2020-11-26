use crate::aten::native::loops;
use crate::c10::*;
use crate::tensor::TensorIterator;
use crate::Closure;
use crate::{AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2, AT_PRIVATE_CASE_TYPE};

pub fn add_kernel(iter: &mut TensorIterator) {
    if iter.dtype() == ScalarType::Bool {
        todo!()
    } else {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2!(iter.dtype(), "add_cpu", || {
            loops::cpu_kernel_vec(
                iter,
                Closure::new(|args: [SCALART; 2]| -> SCALART { args[0] + args[1] }),
            )
        })
    }
}

pub fn div_kernel(iter: &mut TensorIterator) {
    if is_intgeral_type(iter.dtype(), false) {
        todo!()
    } else {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2!(iter.dtype(), "div_cpu", || {
            loops::cpu_kernel_vec(
                iter,
                Closure::new(|args: [SCALART; 2]| -> SCALART { args[0] / args[1] }),
            )
        })
    }
}
pub fn mul_kernel(iter: &mut TensorIterator) {
    if iter.dtype() == ScalarType::Bool {
        // loops::cpu_kernel(iter, [=](bool a, bool b) -> bool { return a && b; });
    } else {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2!(iter.dtype(), "mul_cpu", || {
            loops::cpu_kernel_vec(
                iter,
                Closure::new(|args: [SCALART; 2]| -> SCALART { args[0] * args[1] }),
            )
        })
    }
}
pub fn sub_kernel(iter: &mut TensorIterator) {
    if iter.dtype() == ScalarType::Bool {
        todo!()
    } else {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2!(iter.dtype(), "sub_cpu", || {
            loops::cpu_kernel_vec(
                iter,
                Closure::new(|args: [SCALART; 2]| -> SCALART { args[0] - args[1] }),
            )
        })
    }
}
