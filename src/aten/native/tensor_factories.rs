use std::ptr::copy_nonoverlapping;

use crate::aten::util::prod_intlist;
use crate::c10::{MemoryFormat, Scalar, Storage, StorageImpl, TensorOptions, TypeMeta};
use crate::core::get_cpu_allocator;
use crate::tensor::{Tensor, TensorImpl};
use crate::{
    aten::native, c10::ScalarType, AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2, AT_PRIVATE_CASE_TYPE,
};

pub fn empty_cpu<T: Into<Option<MemoryFormat>>, A: AsRef<TensorOptions>>(
    size: &[usize],
    options: A,
    _optional_memory_format: T,
) -> Tensor {
    let allocator = unsafe { &mut *get_cpu_allocator() };

    let nelements = prod_intlist(size);
    let options = options.as_ref();
    let dtype = options.dtype();
    let size_bytes = nelements * dtype.itemsize();

    let storage_impl =
        StorageImpl::new(size_bytes, allocator.allocate(size_bytes), allocator, true);

    let tensor = make_tensor(storage_impl, &dtype);
    // Default TensorImpl has size [0]
    if size.len() != 1 || size[0] != 0 {
        tensor.get_unsafe_tensor_impl().set_sizes_contiguous(size);
    }

    tensor
}
pub fn empty_strided_cpu<A: AsRef<TensorOptions>>(
    size: &[usize],
    stride: &[usize],
    options: A,
) -> Tensor {
    // check_size_nonnegative(size);
    let t = empty_cpu(&[0], options, None);
    native::resize_impl_cpu(t.get_unsafe_tensor_impl(), size, Some(stride));
    return t;
}

pub fn empty<T: Into<Option<MemoryFormat>>, A: AsRef<TensorOptions>>(
    size: &[usize],
    options: A,
    optional_memory_format: T,
) -> Tensor {
    let result = empty_cpu(size, options, optional_memory_format);
    result
}

pub fn empty_like<T: Into<Option<MemoryFormat>>, A: AsRef<TensorOptions>>(
    self_: &Tensor,
    options: A,
    optional_memory_format: T,
) -> Tensor {
    let options = self_.options().merge_in(options);
    let result;
    result = empty(self_.sizes(), &options, optional_memory_format);
    result
}

// --------------------- full --------------------- //

pub fn infer_full_options<A: AsRef<TensorOptions>>(
    _fill_value: Scalar,
    options: A,
) -> TensorOptions {
    let options = options.as_ref().clone();
    if !options.has_dtype() {
        let _ = ();
    }
    options
}

pub fn full<A: AsRef<TensorOptions>>(
    size: &[usize],
    fill_value: impl Into<Scalar>,
    options: A,
) -> Tensor {
    let fill_value: Scalar = fill_value.into();
    let result = empty(size, infer_full_options(fill_value, options), None);
    result.fill_(fill_value);
    result
}

pub fn full_out<'a>(result: &'a Tensor, size: &[usize], fill_value: f32) {
    result.resize(size, None);
    result.fill_(fill_value)
}

pub fn full_like<T: Into<Option<MemoryFormat>>, A: AsRef<TensorOptions>>(
    self_: &Tensor,
    fill_value: f32,
    options: A,
    optional_memory_format: T,
) -> Tensor {
    let result = empty_like(self_, options, optional_memory_format);
    result.fill_(fill_value);
    result
}

// ----------------- ones ----------------------- //
pub fn ones<A: AsRef<TensorOptions>>(size: &[usize], options: A) -> Tensor {
    full(size, 1.0, options)
}

pub fn ones_out<'a>(result: &'a Tensor, size: &[usize]) {
    full_out(result, size, 1.)
}

pub fn ones_like<T: Into<Option<MemoryFormat>>, A: AsRef<TensorOptions>>(
    self_: &Tensor,
    options: A,
    optional_memory_format: T,
) -> Tensor {
    let result = empty_like(self_, options, optional_memory_format);
    result.fill_(1.0);
    result
}

pub fn make_tensor<T: Into<Storage>>(storage: T, dtype: &TypeMeta) -> Tensor {
    let storage = storage.into();
    let device = storage.device();
    let impl_ = TensorImpl::new(storage, dtype.clone(), Some(device));
    Tensor::from_impl(impl_)
}

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Scalar Tensor ~~~~~~~~~~~~~~~~~~~~~~~~~~ */
pub fn scalar_tensor<A: AsRef<TensorOptions>>(s: Scalar, options: A) -> Tensor {
    let options = options.as_ref();
    if options.device().is_cpu() {
        let result = empty_cpu(&[], options, None);
        result.fill_(s);
        return result;
    }
    let result = empty(&[], options, None);
    result.fill_(s);
    result
}

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tensor from slice ~~~~~~~~~~~~~~~~~~~~~~~*/

fn tensor_cpu<T, A: AsRef<TensorOptions>>(values: &[T], options: A) -> Tensor {
    let result = empty(&[values.len()], options, None);
    assert!(result.is_contiguous());
    tensor_cpu_kernel(&result, values);
    result
}

fn tensor_cpu_kernel<T>(result: &Tensor, values: &[T]) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2!(result.scalar_type(), "tensor_cpu", || {
        unsafe {
            copy_nonoverlapping(
                values.as_ptr() as *mut SCALART,
                result.data_ptr().as_ptr() as *mut SCALART,
                values.len(),
            );
        }
    });
}

pub fn tensor<T, A: AsRef<TensorOptions>>(values: &[T], options: A) -> Tensor {
    let options = options.as_ref();
    if options.device().is_cpu() {
        tensor_cpu(values, options)
    } else {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_empty_cpu_default() {
        crate::init_rovo();
        let size = [2usize; 2];
        let tensor = empty(&size, TensorOptions::default(), None);
        println!("{:?}", tensor);
    }
    #[test]
    fn test_fill() {
        crate::init_rovo();
        let options = TensorOptions::default();
        let size = [2usize; 2];
        let tensor = empty_cpu(&size, &options, None);
        tensor.fill_(1.0);
        println!("{:?}", tensor);
        let tensor = empty_cpu(&size, &options, None);
        tensor.fill_(1.4);
        println!("{:?}", tensor);
    }
}
