use super::native;
use crate::tensor::Tensor;

pub fn as_strided<'b, 'c>(
    self_: &Tensor,
    size: impl Into<&'b [usize]>,
    strides: impl Into<&'c [usize]>,
    storage_offset: Option<usize>,
) -> Tensor {
    native::as_strided_tensorimpl(self_, size.into(), strides.into(), storage_offset)
}

pub fn mm(mat1: &Tensor, mat2: &Tensor)->Tensor{
    native::mm_cpu(mat1, mat2)
}