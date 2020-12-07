use crate::tensor::Tensor;

pub fn expand_size(to_expand: &Tensor, sizes: &[usize]) -> Tensor {
    if to_expand.sizes() == sizes {
        return to_expand.clone();
    }
    to_expand.expand(sizes, true)
}

pub fn infer_expand_geometry(
    tensor_sizes: &[usize],
    tensor_strides: &[usize],
    sizes: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    let ndim = sizes.len();
    let tensor_dim = tensor_sizes.len();
    if tensor_dim == 0 {
        return (sizes.into(), vec![0; ndim]);
    }
    let mut expanded_sizes = vec![0; ndim];
    let mut expanded_strides = vec![0; ndim];
    for i in (0..ndim as isize).rev() {
        let offset = ndim as isize - 1 - i;
        let dim = tensor_dim as isize - 1 - offset;
        let mut size = if dim >= 0 {
            tensor_sizes[dim as usize]
        } else {
            1
        };
        let mut stride = if dim >= 0 {
            tensor_strides[dim as usize]
        } else {
            expanded_sizes[i as usize + 1] * expanded_strides[i as usize + 1]
        };
        let target_size = sizes[i as usize];
        if size != target_size {
            assert!(size == 1);
            size = target_size;
            stride = 0;
        }
        expanded_sizes[i as usize] = size;
        expanded_strides[i as usize] = stride;
    }
    (expanded_sizes, expanded_strides)
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
    tensor
}