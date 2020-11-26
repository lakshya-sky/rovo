use crate::{
    aten::native::cpu_kernel_vec,
    c10::{DeviceType, ScalarType},
    tensor::TensorIterator,
    Closure, AT_DISPATCH_ALL_TYPES_AND, AT_PRIVATE_CASE_TYPE,
};

pub fn copy_kernel(_: DeviceType, iter: &mut TensorIterator, non_blocking: bool) {
    let dtype = iter.dtype_(0);

    if dtype == iter.dtype_(1) {
        AT_DISPATCH_ALL_TYPES_AND!(dtype, "copy_kernel", || {
            cpu_kernel_vec(
                iter,
                Closure::new(|args: [SCALART; 1]| -> SCALART { args[0] }),
            )
        });
    } else {
    }
}
