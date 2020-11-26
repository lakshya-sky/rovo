use crate::c10::MemoryFormat;
use crate::tensor::Tensor;
pub fn contiguous(self_: &Tensor) -> Tensor {
    contiguous_(self_, MemoryFormat::Contiguous)
}
pub fn contiguous_(self_: &Tensor, memory_format: MemoryFormat) -> Tensor {
    if self_.is_contiguous_(memory_format) {
        return self_.clone();
    }
    let result = super::empty_like(self_, self_.options(), memory_format);
    result.copy(self_, None);
    result
}
