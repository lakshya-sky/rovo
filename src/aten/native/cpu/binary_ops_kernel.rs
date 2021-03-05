use crate::aten::native::loops;
use crate::tensor::TensorIterator;
use crate::Closure;
use crate::AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2;
use crate::{c10::*, AT_DISPATCH_FLOATING_TYPES};

pub fn add_kernel(iter: &mut TensorIterator, alpha: Scalar) {
    if iter.dtype() == ScalarType::Bool {
        todo!()
    } else {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2!(_, _, iter.dtype(), "add_cpu/sub_cpu", || {
            let alpha: Scalart = alpha.to();
            loops::cpu_kernel_vec(
                iter,
                Closure::new(|args: [Scalart; 2]| -> Scalart { args[0] + alpha * args[1] }),
            )
        })
    }
}

pub fn div_kernel(iter: &mut TensorIterator) {
    if isIntegralType(iter.dtype(), false) {
        todo!()
    } else {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2!(_, _, iter.dtype(), "div_cpu", || {
            loops::cpu_kernel_vec(
                iter,
                Closure::new(|args: [Scalart; 2]| -> Scalart { args[0] / args[1] }),
            )
        })
    }
}

pub fn mul_kernel(iter: &mut TensorIterator) {
    if iter.dtype() == ScalarType::Bool {
        // loops::cpu_kernel(iter, [=](bool a, bool b) -> bool { return a && b; });
    } else {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2!(_, _, iter.dtype(), "mul_cpu", || {
            loops::cpu_kernel_vec(
                iter,
                Closure::new(|args: [Scalart; 2]| -> Scalart { args[0] * args[1] }),
            )
        })
    }
}

pub fn sub_kernel(iter: &mut TensorIterator) {
    if iter.dtype() == ScalarType::Bool {
        todo!()
    } else {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2!(_, _, iter.dtype(), "sub_cpu", || {
            loops::cpu_kernel_vec(
                iter,
                Closure::new(|args: [Scalart; 2]| -> Scalart { args[0] - args[1] }),
            )
        })
    }
}

pub fn sigmoid_backward_kernel(iter: &mut TensorIterator) {
    AT_DISPATCH_FLOATING_TYPES!(iter.dtype(), "sigmoid_backward_cpu", || {
        loops::cpu_kernel_vec(
            iter,
            Closure::new(|args: [Scalart; 2]| -> Scalart {
                args[0] * (1 as Scalart - args[1]) * args[1]
            }),
        )
    })
}
