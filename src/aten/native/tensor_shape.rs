use aten::{computeStride, infer_size};

use super::make_tensor;
use crate::{aten, tensor::TensorImpl};
use crate::{
    c10::Storage,
    tensor::{maybe_wrap_dim, Tensor},
};

pub fn expand(self_: &Tensor, size: &[usize], _implicit: bool) -> Tensor {
    let (expandedSizes, expandedStrides) =
        aten::infer_expand_geometry(self_.sizes(), self_.strides(), size);
    let result = self_.as_strided(expandedSizes.as_slice(), expandedStrides.as_slice());
    result
}

pub fn setStrided(self_: &Tensor, size: &[usize], strides: &[usize], storage_offset: usize) {
    let self_ = self_.get_unsafe_tensor_impl();
    self_.set_storage_offset(storage_offset);
    if self_.sizes() == size && self_.strides() == strides {
        return;
    }
    self_.set_sizes_and_strides(size, strides);
}

pub fn as_strided_tensorimpl<'a>(
    self_: &'a Tensor,
    size: &[usize],
    strides: &[usize],
    storage_offset: Option<usize>,
) -> Tensor {
    let storage_offset = storage_offset.unwrap_or(self_.storage_offset());
    let result = make_tensor(self_.storage(), self_.dtype());
    setStrided(&result, size, strides, storage_offset);
    result
}
pub fn as_strided_<'a, 'b, 'c>(
    self_: &'a Tensor,
    size: impl Into<&'b [usize]>,
    strides: impl Into<&'c [usize]>,
    storage_offset: Option<usize>,
) -> &'a Tensor {
    let storage_offset = storage_offset.unwrap_or(self_.storage_offset());
    setStrided(self_, size.into(), strides.into(), storage_offset);
    self_
}

pub fn t(self_: &Tensor) -> Tensor {
    transpose(self_, 0, if self_.dim() < 2 { 0 } else { 1 })
}
pub fn t_<'a>(self_: &'a Tensor) -> &'a Tensor {
    transpose_(self_, 0, if self_.dim() < 2 { 0 } else { 1 })
}

pub fn transpose(self_: &Tensor, dim0: i64, dim1: i64) -> Tensor {
    let ndims = self_.dim();
    let dim0 = maybe_wrap_dim(dim0, ndims, true);
    let dim1 = maybe_wrap_dim(dim1, ndims, true);
    if dim0 == dim1 {
        return self_.clone();
    }

    let mut strides = self_.strides().to_vec();
    let mut sizes = self_.sizes().to_vec();
    strides.swap(dim0, dim1);
    sizes.swap(dim0, dim1);
    let result = self_.as_strided(sizes.as_slice(), strides.as_slice());
    result
}

pub fn transpose_<'a>(self_: &'a Tensor, dim0: i64, dim1: i64) -> &'a Tensor {
    let ndims = self_.dim();
    let dim0 = maybe_wrap_dim(dim0, ndims, true);
    let dim1 = maybe_wrap_dim(dim1, ndims, true);
    if dim0 == dim1 {
        return self_;
    }

    let mut strides = self_.strides().to_vec();
    let mut sizes = self_.sizes().to_vec();
    strides.swap(dim0, dim1);
    sizes.swap(dim0, dim1);
    self_.as_strided_(sizes.as_slice(), strides.as_slice())
}

fn infer_squeeze_geometry(tensor: &Tensor) -> (Vec<usize>, Vec<usize>) {
    let mut sizes = vec![];
    let mut strides = vec![];

    for d in 0..tensor.dim() as usize {
        if tensor.sizes()[d] != 1 {
            sizes.push(tensor.size(d));
            strides.push(tensor.stride(d));
        }
    }
    (sizes, strides)
}

fn infer_squeeze_geometry_with_dim(tensor: &Tensor, dim: usize) -> (Vec<usize>, Vec<usize>) {
    let mut sizes = vec![];
    let mut strides = vec![];

    for d in 0..tensor.dim() as usize {
        if d != dim || tensor.sizes()[dim] != 1 {
            sizes.push(tensor.size(d));
            strides.push(tensor.stride(d));
        }
    }
    (sizes, strides)
}

pub fn squeeze(self_: &Tensor) -> Tensor {
    let g = infer_squeeze_geometry(self_);
    let result;
    result = self_.as_strided(g.0.as_slice(), g.1.as_slice());
    result
}

fn alias_with_sizes_and_strides(self_: &Tensor, sizes: &[usize], strides: &[usize]) -> Tensor {
    let new_self;
    //   if self_.is_quantized(){
    //     auto impl = c10::make_intrusive<QTensorImpl>(
    //         Storage(self.storage()),
    //         self.key_set(),
    //         self.dtype(),
    //         get_qtensorimpl(self)->quantizer());
    //     impl->set_storage_offset(self.storage_offset());
    //     impl->set_sizes_and_strides(sizes, strides);
    //     self_ = Tensor(std::move(impl));
    //   } else {
    let mut impl_ = TensorImpl::new(self_.storage().clone(), self_.dtype().clone(), None);
    impl_.set_storage_offset(self_.storage_offset());
    impl_.set_sizes_and_strides(sizes, strides);
    new_self = Tensor::from_impl(impl_);
    //   }
    return new_self;
}

pub fn view(self_: &Tensor, size: &[usize]) -> Tensor {
    let inferred_size = infer_size(
        size.iter()
            .map(|i| *i as isize)
            .collect::<Vec<isize>>()
            .as_slice(),
        self_.numel(),
    );
    let stride = computeStride(self_.sizes(), self_.strides(), inferred_size.as_slice());
    assert!(stride.is_some(), "view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.");
    let stride_value = stride.unwrap();
    alias_with_sizes_and_strides(self_, inferred_size.as_slice(), stride_value.as_slice())
}
