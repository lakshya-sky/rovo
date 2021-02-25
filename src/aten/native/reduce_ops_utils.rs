use crate::{
    autograd::empty,
    c10::{self, ScalarType},
    tensor::{maybe_wrap_dim, DimMask, DimVector, Tensor, TensorIterator},
};
pub fn make_dim_mask(dims: &[usize], ndim: i64) -> DimMask {
    let mut mask = DimMask::default();
    if dims.is_empty() {
        mask.flip_all();
    } else {
        for dim in dims {
            let pos_dim = maybe_wrap_dim(*dim as i64, ndim, false);
            assert!(
                pos_dim < 64,
                "reduction operations for dim>=64 are not supported"
            );
            mask.set(pos_dim);
        }
    }
    return mask;
}

pub fn allocate_reduction_result(
    result: &mut Tensor,
    self_: &Tensor,
    mask: &DimMask,
    keepdim: bool,
    dtype: ScalarType,
) {
    let mut shape: DimVector = self_.sizes().into();
    for dim in (0..shape.len()).rev() {
        if mask.check(dim) {
            if keepdim {
                shape[dim] = 1;
            } else {
                shape.remove(dim);
            }
        }
    }
    if result.defined() {
        result.resize(shape.as_slice(), None);
    } else {
        let mut optns = self_.options();
        optns.set_dtype_mut(Some(dtype.into()));
        result.move_tensor(empty(shape.as_slice(), optns, None));
    }
}

pub fn review_reduce_result(result: &Tensor, ndim: i64, mask: DimMask, keepdim: bool) -> Tensor {
    if keepdim {
        return result.clone();
    }
    let mut shape: DimVector = result.sizes().into();
    let mut stride: DimVector = result.strides().into();

    for dim in 0..ndim as usize {
        if mask.check(dim) {
            shape.insert(dim, 1);
            stride.insert(dim, 0);
        }
    }
    return result.as_strided(shape.as_slice(), stride.as_slice());
}

pub fn make_reduction(
    _name: &str,
    result: &mut Tensor,
    self_: &Tensor,
    dim: &[usize],
    keepdim: bool,
    in_dtype: ScalarType,
    out_dtype: ScalarType,
) -> TensorIterator {
    // check that result type and dtype match if provided
    assert!((!result.defined() || result.scalar_type() == out_dtype));

    let ndim = self_.dim();
    let mask = make_dim_mask(dim, ndim);
    allocate_reduction_result(result, self_, &mask, keepdim, out_dtype);
    let viewed_result = review_reduce_result(result, ndim, mask, keepdim);
    if self_.scalar_type() == in_dtype {
        return TensorIterator::reduce_op(&viewed_result, self_);
    }
    return TensorIterator::reduce_op(&viewed_result, &self_.to_dtype(in_dtype));
}

pub fn get_dtype(
    result: &Tensor,
    self_: &Tensor,
    dtype: Option<ScalarType>,
    promote_integers: bool,
) -> ScalarType {
    if let Some(d) = dtype {
        return d;
    } else if result.defined() {
        return result.scalar_type();
    }
    let src_type = self_.scalar_type();
    if promote_integers && c10::isIntegralType(src_type, /*includeBool=*/ true) {
        return c10::kLong;
    }
    return src_type;
}
