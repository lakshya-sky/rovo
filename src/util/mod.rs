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

pub fn is_expandable_to(shape: &[usize], desired: &[usize]) -> bool {
    let shape_dim = shape.len();
    let desired_dim = desired.len();
    if shape_dim > desired_dim {
        return false;
    }
    let mut i = 0usize;
    loop {
        if i >= shape_dim {
            break;
        }
        let size = shape[shape_dim - i - 1];
        let target = desired[desired_dim - i - 1];
        if size != target && size != 1 {
            return false;
        }
        i += 1;
    }
    return true;
}

pub fn sum_to(mut tensor: Tensor, shape: &[usize]) -> Tensor {
    if shape.len() == 0 {
        return tensor.sum();
    }
    let mut reduce_dims = smallvec::SmallVec::<[usize; 8]>::new();
    let sizes = tensor.sizes();
    let leading_dims = sizes.len() - shape.len();
    for i in 0..leading_dims {
        reduce_dims.push(i);
    }
    for i in leading_dims..sizes.len() {
        if shape[i - leading_dims] == 1 && sizes[i] != 1 {
            reduce_dims.push(i);
        }
    }
    if !reduce_dims.is_empty() {
        tensor = tensor.sum_dim(reduce_dims.as_slice(), true)
    }
    println!("sum_to: Tensor shape {:?}", tensor.sizes());
    tensor
}
