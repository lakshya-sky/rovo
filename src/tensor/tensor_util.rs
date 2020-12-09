#[inline(always)]
pub fn maybe_wrap_dim(dim: i64, dim_post_expr: i64, _wrap_scalr: bool) -> usize {
    let mut dim = dim;
    let mut dim_post_expr = dim_post_expr;
    if dim_post_expr <= 0 {
        dim_post_expr = 1;
    }
    let min = -dim_post_expr;
    let max = dim_post_expr - 1;
    assert!(
        dim >= min && dim <= max,
        "Dimensions out of range (expected to be in range of [{},{}] but got {})",
        min,
        max,
        dim
    );

    if dim < 0 {
        dim += dim + dim_post_expr;
    }

    dim as usize
}

#[inline(always)]
pub fn infer_size(a: &[usize], b: &[usize]) -> Vec<usize> {
    // eprintln!("a: {:?}, b: {:?}", a, b);
    let dims_a = a.len();
    let dims_b = b.len();
    let n_dim = if dims_a > dims_b { dims_a } else { dims_b };
    let mut expanded_sizes: Vec<usize> = (0..n_dim).into_iter().collect();
    // eprintln!("ndim: {}", n_dim);
    for i in (0..n_dim as isize).rev() {
        let offset = (n_dim as isize) - 1 - i;
        let dim_a = dims_a as isize - 1 - offset;
        let dim_b = dims_b as isize - 1 - offset;
        let size_a = if dim_a >= 0 { a[dim_a as usize] } else { 1 };
        let size_b = if dim_b >= 0 { b[dim_b as usize] } else { 1 };
        assert!(
            size_a == size_b || size_a == 1 || size_b == 1,
            "The size of tensor a ({}) must match the size of tensor b ({}) at non-singleton dimension {}",
            size_a,
            size_b,
            i
        );

        expanded_sizes[i as usize] = if size_a == 1 { size_b } else { size_a };
    }
    expanded_sizes
}
