#[inline]
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
