use crate::aten::util::prod_intlist;
use crate::c10::{MemoryFormat, Storage, StorageImpl, TypeMeta};
use crate::core::getCPUAllocator;
use crate::tensor::{NewTensor, NewTensorImpl, TensorOptions};
use std::{cell::RefCell, rc::Rc};

pub fn empty_cpu(
    size: &[usize],
    options: &TensorOptions,
    _optional_memory_format: Option<MemoryFormat>,
) -> NewTensor {
    let allocator = unsafe { &mut *getCPUAllocator() };

    let nelements = prod_intlist(size);
    let dtype = options.dtype();
    let size_bytes = nelements * dtype.itemsize();

    let storage_impl =
        StorageImpl::new(size_bytes, allocator.allocate(size_bytes), allocator, true);

    let tensor = make_tensor(storage_impl, dtype);
    // Default TensorImpl has size [0]
    if size.len() != 1 || size[0] != 0 {
        tensor.get_unsafe_tensor_impl().set_sizes_contiguous(size);
    }

    tensor
}

pub fn make_tensor(storage_impl: StorageImpl, dtype: TypeMeta) -> NewTensor {
    let storage = Storage::new(Rc::new(RefCell::new(storage_impl)));
    let device = storage.device();
    let impl_ = NewTensorImpl::new(storage, dtype, Some(device));
    NewTensor::from_impl(impl_)
}

fn full(size: &[usize], options: &TensorOptions, fill_value: f32) -> NewTensor {
    let result = empty_cpu(size, options, None);
    let _ = result.fill_(fill_value);
    result
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_empty_cpu_default() {
        crate::init_rovo();
        let options = TensorOptions::default();
        let size = [2usize; 2];
        let tensor = empty_cpu(&size, &options, None);
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
