use super::*;
pub fn select(self_: &NewTensor, dim: i64, mut index: i64) -> NewTensor {
    let ndim = self_.dim();
    if ndim == 0 {
        panic!("Select can't be applied to a 0-dim index");
    }
    let dim = maybe_wrap_dim(dim, ndim, true);
    let size = self_.size(dim);
    if index < -(size as i64) || index >= (size as i64) {
        panic!();
    }
    if index < 0 {
        index += size as i64;
    }
    let index = index as usize;
    let mut sizes = self_.sizes().to_vec();
    let mut strides = self_.strides().to_vec();
    let storage_offset = self_.storage_offset() + index * strides[dim];
    sizes.remove(dim);
    strides.remove(dim);
    dbg!(&sizes, &strides);
    let result = as_strided_tensorimpl(
        self_,
        sizes.as_slice(),
        strides.as_slice(),
        Some(storage_offset),
    );
    result
}
pub fn as_strided_tensorimpl(
    self_: &NewTensor,
    size: &[usize],
    stride: &[usize],
    storage_offset: Option<usize>,
) -> NewTensor {
    let storage_offset = storage_offset.unwrap_or_else(|| self_.storage_offset());
    let result = crate::aten::native::tensor_factories::make_tensor(self_.storage(), self_.dtype());
    set_strided(&result, size, stride, storage_offset);
    result
}

pub fn set_strided(self_: &NewTensor, size: &[usize], stride: &[usize], storage_offset: usize) {
    let self_ = self_.get_unsafe_tensor_impl();
    self_.set_storage_offset(storage_offset);
    self_.set_sizes_and_strides(size, stride);
}
