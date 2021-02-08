use crate::c10::{DataPtr, MemoryFormat, StorageImpl};
use crate::tensor::{Tensor, TensorImpl};
pub fn storage_size_for(size: &[usize], stride: &[usize]) -> usize {
    debug_assert_eq!(size.len(), stride.len(), "storage_size_for(size, stride) requires that size and stride have the same size as a precondition.");
    let mut storage_size = 1;
    for dim in 0..size.len() {
        if size[dim] == 0 {
            storage_size = 0;
            break;
        }
        storage_size += (size[dim] - 1) * stride[dim];
    }
    storage_size
}

pub fn resize<'a>(
    self_: &'a Tensor,
    size: &[usize],
    _optional_memory_format: Option<crate::c10::MemoryFormat>,
) -> &'a Tensor {
    let stride = None;
    let impl_ = self_.get_unsafe_tensor_impl();
    resize_impl_cpu(impl_, size, stride);
    self_
}

pub fn resize_impl_cpu(self_: &mut TensorImpl, size: &[usize], stride: Option<&[usize]>) {
    let storage_size;
    if let Some(stride_) = stride {
        self_.set_sizes_and_strides(size, stride_);
        storage_size = storage_size_for(size, stride_);
    } else {
        self_.set_sizes_contiguous(size);
        storage_size = self_.numel();
    }
    may_be_resize_storage_cpu(self_, storage_size);
}

pub fn may_be_resize_storage_cpu(self_: &mut TensorImpl, new_size: usize) {
    if new_size > 0 {
        let new_size_bytes = (new_size + self_.storage_offset()) * self_.dtype().itemsize();
        if new_size_bytes > self_.storage().nbytes() {
            resize_bytes(get_storage_ptr(self_), new_size_bytes);
        }
    }
}

pub fn get_storage_ptr(self_: &TensorImpl) -> &mut StorageImpl {
    &mut *self_.storage().get_unsafe_storage_impl()
}

pub fn resize_bytes(storage: &mut StorageImpl, size_bytes: usize) {
    if storage.resizable() {
        let mut new_data = DataPtr::default();
        if size_bytes != 0 {
            new_data = storage.allocator().allocate(size_bytes);
        }
        let old_data = storage.set_data_ptr(new_data);
        let old_capacity = storage.nbytes();
        storage.set_nbytes(size_bytes);
        if !old_data.is_empty() {
            let mut copy_capacity = old_capacity;
            if storage.nbytes() < copy_capacity {
                copy_capacity = storage.nbytes();
            }
            if copy_capacity > 0 {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        old_data.get().unwrap(),
                        storage.data().unwrap(),
                        copy_capacity,
                    )
                }
            }
        }
    } else {
        panic!("Trying to resize storage that is not resizable")
    }
}

pub fn resize_as_<'a>(
    self_: &'a Tensor,
    other: &Tensor,
    optional_memory_format: Option<MemoryFormat>,
) -> &'a Tensor {
    let result = self_.resize(other.sizes(), None);
    if let Some(mut m) = optional_memory_format {
        if m == MemoryFormat::Preserve {
            m = other.suggest_memory_format(false);
        }
        self_.get_unsafe_tensor_impl().empty_tensor_restride(m);
    }
    result
}
