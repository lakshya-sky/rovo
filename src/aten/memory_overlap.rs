use crate::c10::K_STRIDED;
use crate::tensor::{Tensor, TensorImpl};

#[derive(PartialEq)]
pub enum MemOverlap {
    No,
    Yes,
    TooHard,
}

#[derive(PartialEq)]
pub enum MemOverlapStatus {
    Full,
    Partial,
    No,
    TooHard,
}

pub fn has_internal_overlap(tensor: &Tensor) -> MemOverlap {
    has_internal_overlap_(tensor.get_unsafe_tensor_impl())
}

pub fn has_internal_overlap_(t: &TensorImpl) -> MemOverlap {
    assert!(t.layout() == K_STRIDED);

    if t.is_contiguous() {
        return MemOverlap::No;
    }

    let strides = t.strides();
    let sizes = t.sizes();
    for i in 0..strides.len() {
        if strides[i] == 0 && sizes[i] > 1 {
            return MemOverlap::Yes;
        }
    }

    return MemOverlap::TooHard;
}

pub fn assert_no_internal_overlap(tensor: &Tensor) {
    assert_no_internal_overlap_(tensor.get_unsafe_tensor_impl());
}

pub fn assert_no_internal_overlap_(t: &TensorImpl) {
    assert!(
        has_internal_overlap_(t) != MemOverlap::Yes,
        "unsupported operation: more than one element of the written-to tensor 
      refers to a single memory location. Please clone() the tensor before 
      performing the operation."
    );
}

pub fn get_overlap_status(a: &Tensor, b: &Tensor) -> MemOverlapStatus {
    return get_overlap_status_(a.get_unsafe_tensor_impl(), b.get_unsafe_tensor_impl());
}

pub fn get_overlap_status_(a: &TensorImpl, b: &TensorImpl) -> MemOverlapStatus {
    if a as *const TensorImpl == b as *const TensorImpl {
        return MemOverlapStatus::Full;
    }
    if a.numel() == 0 || b.numel() == 0 {
        return MemOverlapStatus::No;
    }
    if !a.is_contiguous() || !b.is_contiguous() {
        return MemOverlapStatus::TooHard;
    }

    if a.storage().data::<u8>() == b.storage().data::<u8>() {
        unsafe {
            let a_begin = a.data().as_ptr() as *const u8;
            let a_end = a_begin.offset((a.numel() * a.itemsize()) as isize);
            let b_begin = b.data().as_ptr() as *const u8;
            let b_end = b_begin.offset((b.numel() * b.itemsize()) as isize);

            if a_begin == b_begin && a_end == b_end {
                return MemOverlapStatus::Full;
            }
            if a_begin < b_end && b_begin < a_end {
                return MemOverlapStatus::Partial;
            }
        }
    }
    return MemOverlapStatus::No;
}

pub fn assert_no_partial_overlap(a: &Tensor, b: &Tensor) {
    assert_no_partial_overlap_(a.get_unsafe_tensor_impl(), b.get_unsafe_tensor_impl())
}
pub fn assert_no_partial_overlap_(a: &TensorImpl, b: &TensorImpl) {
    assert!(
        get_overlap_status_(a, b) != MemOverlapStatus::Partial,
        "unsupported operation: some elements of the input tensor and
        the written-to tensor refer to a single memory location. 
        Please clone() the tensor before performing the operation."
    )
}
