use crate::c10::K_STRIDED;
use crate::tensor::{NewTensor, NewTensorImpl};

#[derive(PartialEq)]
pub enum MemOverlap {
    NO,
    YES,
    TOOHARD,
}

#[derive(PartialEq)]
pub enum MemOverlapStatus {
    FULL,
    PARTIAL,
    NO,
    TOOHARD,
}

pub fn has_internal_overlap(tensor: &NewTensor) -> MemOverlap {
    has_internal_overlap_(tensor.get_unsafe_tensor_impl())
}

pub fn has_internal_overlap_(t: &NewTensorImpl) -> MemOverlap {
    assert!(t.layout() == K_STRIDED);

    if t.is_contiguous() {
        return MemOverlap::NO;
    }

    let strides = t.strides();
    let sizes = t.sizes();
    for i in 0..strides.len() {
        if strides[i] == 0 && sizes[i] > 1 {
            return MemOverlap::YES;
        }
    }

    return MemOverlap::TOOHARD;
}

pub fn assert_no_internal_overlap(tensor: &NewTensor) {
    assert_no_internal_overlap_(tensor.get_unsafe_tensor_impl());
}

pub fn assert_no_internal_overlap_(t: &NewTensorImpl) {
    assert!(
        has_internal_overlap_(t) != MemOverlap::YES,
        "unsupported operation: more than one element of the written-to tensor 
      refers to a single memory location. Please clone() the tensor before 
      performing the operation."
    );
}

pub fn get_overlap_status(a: &NewTensor, b: &NewTensor) -> MemOverlapStatus {
    return get_overlap_status_(a.get_unsafe_tensor_impl(), b.get_unsafe_tensor_impl());
}

pub fn get_overlap_status_(a: &NewTensorImpl, b: &NewTensorImpl) -> MemOverlapStatus {
    if a as *const NewTensorImpl == b as *const NewTensorImpl {
        return MemOverlapStatus::FULL;
    }
    if a.numel() == 0 || b.numel() == 0 {
        return MemOverlapStatus::NO;
    }
    if !a.is_contiguous() || !b.is_contiguous() {
        return MemOverlapStatus::TOOHARD;
    }

    if a.storage().data::<u8>() == b.storage().data::<u8>() {
        unsafe {
            let a_begin = a.data().as_ptr() as *const u8;
            let a_end = a_begin.offset((a.numel() * a.itemsize()) as isize);
            let b_begin = b.data().as_ptr() as *const u8;
            let b_end = b_begin.offset((b.numel() * b.itemsize()) as isize);

            if a_begin == b_begin && a_end == b_end {
                return MemOverlapStatus::FULL;
            }
            if a_begin < b_end && b_begin < a_end {
                return MemOverlapStatus::PARTIAL;
            }
        }
    }
    return MemOverlapStatus::NO;
}

pub fn assert_no_partial_overlap(a: &NewTensor, b: &NewTensor) {
    assert_no_partial_overlap_(a.get_unsafe_tensor_impl(), b.get_unsafe_tensor_impl())
}
pub fn assert_no_partial_overlap_(a: &NewTensorImpl, b: &NewTensorImpl) {
    assert!(
        get_overlap_status_(a, b) != MemOverlapStatus::PARTIAL,
        "unsupported operation: some elements of the input tensor and
        the written-to tensor refer to a single memory location. 
        Please clone() the tensor before performing the operation."
    )
}
