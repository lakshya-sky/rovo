#[inline(always)]
pub fn infer_size(shape: &[isize], numel: usize) -> Vec<usize> {
    let mut res = shape.to_vec();
    let mut newsize = 1;
    let mut infer_dim = None;
    let mut dim = 0;
    let ndim = shape.len();
    while dim != ndim {
        if shape[dim] == -1 {
            if infer_dim.is_some() {
                panic!("only one dimension can be inferred");
            }
            infer_dim = Some(dim);
        } else if shape[dim] >= 0 {
            newsize *= shape[dim];
        } else {
            panic!("invalid shape dimension {}", shape[dim]);
        }
        dim += 1;
    }
    if numel as isize == newsize
        || (infer_dim.is_some() && newsize > 0 && (numel as isize % newsize == 0))
    {
        if infer_dim.is_some() {
            // We have a degree of freedom here to select the dimension size; follow
            // NumPy semantics and just bail.  However, a nice error message is needed
            // because users often use `view` as a way to flatten & unflatten
            // dimensions and will otherwise be confused why
            //   empty_tensor.view( 0, 0)
            // works yet
            //   empty_tensor.view(-1, 0)
            // doesn't.
            assert!(newsize != 0, "cannot reshape tensor of 0 elements into shape {:?} because the unspecified dimension size -1 can be any value and is ambiguous", shape);
            res[infer_dim.unwrap()] = numel as isize / newsize;
        }
        let res = res.drain(..).map(|i| i as usize).collect();
        return res;
    }

    panic!(
        "shape '{:?}' is invalid for input of size {} ",
        shape, numel
    );
}
