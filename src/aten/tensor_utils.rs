// On a high level,
// 1. separate `oldshape` into chunks of dimensions, where the dimensions are
//    ``contiguous'' in each chunk, i.e., oldstride[i] = oldshape[i+1] *
//     oldstride[i+1]
// 2. `newshape` must be able to be separated into same number of chunks as
//    `oldshape` was separated into, where each chunk of newshape has matching
//    ``numel'', i.e., number of subspaces, as the corresponding chunk of
//    `oldshape`.
pub fn computeStride(
    oldshape: &[usize],
    oldstride: &[usize],
    newshape: &[usize],
) -> Option<Vec<usize>> {
    if oldshape.is_empty() {
        return Some(vec![1; newshape.len()]);
    }

    // NOTE: stride is arbitrary in the numel() == 0 case;
    // to match NumPy behavior we copy the strides if the size matches, otherwise
    // we use the stride as if it were computed via resize.
    // This could perhaps be combined with the below code, but the complexity
    // didn't seem worth it.
    let numel: usize = oldshape.iter().product();
    if numel == 0 && oldshape == newshape {
        return Some(oldstride.to_vec());
    }

    let mut newstride = vec![0; newshape.len()];
    if numel == 0 {
        for view_d in (0..newshape.len()).rev() {
            if view_d == newshape.len() - 1 {
                newstride[view_d] = 1;
            } else {
                newstride[view_d] = newshape[view_d + 1].max(1) * newstride[view_d + 1];
            }
        }
        return Some(newstride);
    }

    let mut view_d = newshape.len() as isize - 1;
    // stride for each subspace in the chunk
    let mut chunk_base_stride = *oldstride.last().unwrap();
    // numel in current chunk
    let mut tensor_numel = 1;
    let mut view_numel = 1;
    for tensor_d in (0..oldshape.len()).rev() {
        tensor_numel *= oldshape[tensor_d];
        // if end of tensor size chunk, check view
        if tensor_d == 0
            || (oldshape[tensor_d - 1] != 1
                && oldstride[tensor_d - 1] != tensor_numel * chunk_base_stride)
        {
            while view_d >= 0 && (view_numel < tensor_numel || newshape[view_d as usize] == 1) {
                newstride[view_d as usize] = view_numel * chunk_base_stride;
                view_numel *= newshape[view_d as usize];
                view_d -= 1;
            }
            if view_numel != tensor_numel {
                return None;
            }
            if tensor_d > 0 {
                chunk_base_stride = oldstride[tensor_d - 1];
                tensor_numel = 1;
                view_numel = 1;
            }
        }
    }
    if view_d != -1 {
        return None;
    }
    return Some(newstride);
}
