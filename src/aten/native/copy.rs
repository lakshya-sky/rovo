use crate::{
    c10::KCPU,
    tensor::{Tensor, TensorIteratorConfig},
};
fn copy_transpose_valid(self_: &Tensor, src: &Tensor) -> bool {
    const MIN_SZ: usize = 60 * 60;
    return self_.is_contiguous()
        && src.numel() != 0
        && src.dim() == 2
        && src.stride(0) == 1
        && src.stride(1) == src.size(0)
        && self_.scalar_type() == src.scalar_type()
        && self_.numel() >= MIN_SZ;
}
fn copy_same_type_transpose_(_self_: &Tensor, _src: &Tensor) {
    todo!()
}

fn copy_impl<'a>(self_: &'a Tensor, src: &Tensor, non_blocking: bool) -> &'a Tensor {
    if self_.is_same(src) {
        return self_;
    }
    let mut iter = TensorIteratorConfig::default()
        .set_check_mem_overlap(true)
        .add_output(self_)
        .add_input(src)
        .resize_outputs(false)
        .check_all_same_dtype(false)
        .build();
    if iter.numel() == 0 {
        return self_;
    }
    //Todo: Handle Device's logic here, and also quantized and complex tensors
    let device_type = iter.device_type(0);
    if device_type == KCPU && copy_transpose_valid(self_, src) {
        copy_same_type_transpose_(self_, src);
        return self_;
    }
    super::cpu::copy_kernel(device_type, &mut iter, non_blocking);
    self_
}

pub fn copy_<'a>(self_: &'a Tensor, src: &Tensor, non_blocking: bool) -> &'a Tensor {
    copy_impl(self_, src, non_blocking);
    return self_;
}
