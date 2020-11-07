use crate::aten::util::prod_intlist;
use crate::c10::{MemoryFormat, Storage, StorageImpl, TypeMeta};
use crate::core::get_cpu_allocator;
use crate::tensor::{NewTensor, NewTensorImpl, TensorOptions};

pub fn empty_cpu<T: Into<Option<MemoryFormat>>, A: AsRef<TensorOptions>>(
    size: &[usize],
    options: A,
    _optional_memory_format: T,
) -> NewTensor {
    let allocator = unsafe { &mut *get_cpu_allocator() };

    let nelements = prod_intlist(size);
    let options = options.as_ref();
    let dtype = options.dtype();
    let size_bytes = nelements * dtype.itemsize();

    let storage_impl =
        StorageImpl::new(size_bytes, allocator.allocate(size_bytes), allocator, true);

    let tensor = make_tensor(storage_impl, &dtype);
    // Default NewTensorImpl has size [0]
    if size.len() != 1 || size[0] != 0 {
        tensor.get_unsafe_tensor_impl().set_sizes_contiguous(size);
    }

    tensor
}

pub fn empty<T: Into<Option<MemoryFormat>>, A: AsRef<TensorOptions>>(
    size: &[usize],
    options: A,
    optional_memory_format: T,
) -> NewTensor {
    let result = empty_cpu(size, options, optional_memory_format);
    result
}

pub fn empty_like<T: Into<Option<MemoryFormat>>, A: AsRef<TensorOptions>>(
    self_: &NewTensor,
    options: A,
    optional_memory_format: T,
) -> NewTensor {
    let options = self_.options().merge_in(options);
    let result;
    result = empty(self_.sizes(), &options, optional_memory_format);
    result
}

// --------------------- full --------------------- //

pub fn infer_full_options<A: AsRef<TensorOptions>>(_fill_value: f32, options: A) -> TensorOptions {
    let options = options.as_ref().clone();
    if !options.has_dtype() {
        let _ = ();
    }
    options
}

pub fn full<A: AsRef<TensorOptions>>(size: &[usize], fill_value: f32, options: A) -> NewTensor {
    let result = empty(size, infer_full_options(fill_value, options), None);
    result.fill_(fill_value);
    result
}

pub fn full_out<'a>(result: &'a NewTensor, size: &[usize], fill_value: f32) -> &'a NewTensor {
    result.resize(size, None);
    result.fill_(fill_value)
}

pub fn full_like<T: Into<Option<MemoryFormat>>, A: AsRef<TensorOptions>>(
    self_: &NewTensor,
    fill_value: f32,
    options: A,
    optional_memory_format: T,
) -> NewTensor {
    let result = empty_like(self_, options, optional_memory_format);
    result.fill_(fill_value);
    result
}

// ----------------- ones ----------------------- //
pub fn ones<A: AsRef<TensorOptions>>(size: &[usize], options: A) -> NewTensor {
    full(size, 1.0, options)
}

pub fn ones_out<'a>(result: &'a NewTensor, size: &[usize]) -> &'a NewTensor {
    full_out(result, size, 1.)
}

pub fn ones_like<T: Into<Option<MemoryFormat>>, A: AsRef<TensorOptions>>(
    self_: &NewTensor,
    options: A,
    optional_memory_format: T,
) -> NewTensor {
    let result = empty_like(self_, options, optional_memory_format);
    result.fill_(1.0);
    result
}

pub fn make_tensor<T: Into<Storage>>(storage: T, dtype: &TypeMeta) -> NewTensor {
    let storage = storage.into();
    let device = storage.device();
    let impl_ = NewTensorImpl::new(storage, dtype.clone(), Some(device));
    NewTensor::from_impl(impl_)
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
