mod ordered_dict;
use crate::tensor::Tensor;
pub use ordered_dict::*;

pub fn shallow_compare(lhs: Tensor, rhs: Tensor) -> bool {
    if lhs.dim() != rhs.dim() {
        false
    } else {
        let lhs_slice = lhs.get_tensor_impl().data.as_slice();
        let rhs_slice = rhs.get_tensor_impl().data.as_slice();
        if lhs_slice == rhs_slice {
            true
        } else {
            false
        }
    }
}

