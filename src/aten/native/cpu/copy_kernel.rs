use crate::{
    aten::native::{cpu_kernel, cpu_kernel_vec},
    c10::{cast_with_inter_type, DeviceType},
    tensor::TensorIterator,
    Closure, AT_DISPATCH_ALL_TYPES_AND, AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2,
};

pub fn copy_kernel(_: DeviceType, iter: &mut TensorIterator, _non_blocking: bool) {
    let dtype = iter.dtype_(0);
    if dtype == iter.dtype_(1) {
        AT_DISPATCH_ALL_TYPES_AND!(dtype, "copy_kernel", || {
            cpu_kernel_vec(
                iter,
                Closure::new(|args: [SCALART; 1]| -> SCALART { args[0] }),
            )
        });
    } else {
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2!(_, _, dtype, "copy_", || {
            type DEST = SCALART;
            AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2!(_, _, iter.dtype_(1), "copy_", || {
                let closure = Closure::new(|src: [SCALART; 1]| -> DEST {
                    cast_with_inter_type::<SCALART, DEST>(src[0])
                });
                cpu_kernel(iter, closure);
            })
        })
    }
}
