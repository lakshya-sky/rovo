use crate::aten::native::loops;
use crate::c10::*;
use crate::tensor::NewTensorIterator;
use crate::Closure;
use crate::{AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2, AT_PRIVATE_CASE_TYPE};

pub fn add_kernel(iter: &mut NewTensorIterator) {
    if iter.dtype() == ScalarType::Bool {
        todo!()
    } else {
        // ScalarType::BFloat16, ScalarType::Half, iter.dtype(), "add_cpu/sub_cpu", ||{
        //    loops::cpu_kernel_vec(iter, |a: scalar_t,b: scalar_t|->scalar_t{
        //        a+b
        //    });
        // }
        let type_ = iter.dtype();
        // let mut closure = match type_ {
        //     ScalarType::Float => {
        //         type SCALART = f32;
        //         || {
        //             loops::cpu_kernel_vec(
        //                 iter,
        //                 Closure::new(|args: [SCALART; 2]| -> SCALART { args[0] + args[1] }),
        //             )
        //         }
        //     }
        //     _ => todo!(),
        // };
        // closure();
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2!(type_, "add_cpu", || {
            loops::cpu_kernel_vec(
                iter,
                Closure::new(|args: [SCALART; 2]| -> SCALART { args[0] + args[1] }),
            )
        })
    }
}

pub fn div_kernel(iter: &mut NewTensorIterator) {
    if is_intgeral_type(iter.dtype(), false) {
        todo!()
    } else {
        let type_ = iter.dtype();
        let mut closure = match type_ {
            ScalarType::Float => {
                type SCALART = f32;
                || {
                    loops::cpu_kernel_vec(
                        iter,
                        Closure::new(|args: [SCALART; 2]| -> SCALART { args[0] / args[1] }),
                    )
                }
            }
            _ => todo!(),
        };
        closure();
    }
}
pub fn mul_kernel(iter: &mut NewTensorIterator) {
    if iter.dtype() == ScalarType::Bool {
        // loops::cpu_kernel(iter, [=](bool a, bool b) -> bool { return a && b; });
    } else {
        // ScalarType::BFloat16, ScalarType::Half, iter.dtype(), "add_cpu/sub_cpu", ||{
        //    loops::cpu_kernel_vec(iter, |a: scalar_t,b: scalar_t|->scalar_t{
        //        a+b
        //    });
        // }
        let type_ = iter.dtype();
        let mut closure = match type_ {
            ScalarType::Float => {
                type SCALART = f32;
                || {
                    loops::cpu_kernel_vec(
                        iter,
                        Closure::new(|args: [SCALART; 2]| -> SCALART { args[0] * args[1] }),
                    )
                }
            }
            _ => todo!(),
        };
        closure();
    }
}
pub fn sub_kernel(iter: &mut NewTensorIterator) {
    if iter.dtype() == ScalarType::Bool {
        todo!()
    } else {
        // ScalarType::BFloat16, ScalarType::Half, iter.dtype(), "add_cpu/sub_cpu", ||{
        //    loops::cpu_kernel_vec(iter, |a: scalar_t,b: scalar_t|->scalar_t{
        //        a+b
        //    });
        // }
        let type_ = iter.dtype();
        let mut closure = match type_ {
            ScalarType::Float => {
                type SCALART = f32;
                || {
                    loops::cpu_kernel_vec(
                        iter,
                        Closure::new(|args: [SCALART; 2]| -> SCALART { args[0] - args[1] }),
                    )
                }
            }
            _ => todo!(),
        };
        closure();
    }
}
