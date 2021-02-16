use c10::isFloatingType;

use crate::{
    autograd::empty,
    c10::{self, ScalarType},
    tensor::{maybe_wrap_dim, DimMask, DimVector, Tensor, TensorIterator},
};

use super::cpu::{mean_kernel_impl, sum_kernel_impl};

fn make_dim_mask(dims: &[usize], ndim: i64) -> DimMask {
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

fn allocate_reduction_result(
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

fn review_reduce_result(result: &Tensor, ndim: i64, mask: DimMask, keepdim: bool) -> Tensor {
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

fn make_reduction(
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

pub fn mean_out_cpu_gpu<'result>(
    result: &'result mut Tensor,
    self_: &Tensor,
    dim: &[usize],
    keepdim: bool,
    opt_dtype: Option<ScalarType>,
) -> &'result Tensor {
    let scalar_type = if let Some(d) = opt_dtype {
        d
    } else {
        self_.scalar_type()
    };

    assert!(
        isFloatingType(scalar_type),
        "Can only calcualte the mean of floating types. Got {:?} instead",
        scalar_type
    );
    let dtype = get_dtype(result, self_, opt_dtype, true);
    if self_.device().is_cpu() {
        let self_sizes = self_.sizes();
        let mut dim_prod = 1;
        if dim.len() == 0 || self_.dim() == 0 {
            dim_prod = self_.numel();
        } else {
            for d in dim {
                dim_prod *= self_sizes[*d];
            }
        }
        sum_out(result, self_, dim.into(), keepdim, dtype.into());
        result.div_scalar(dim_prod);
        return result;
    }
    let iter = make_reduction("mean", result, self_, dim, keepdim, dtype, dtype);
    if iter.numel() == 0 {
        result.fill_(f64::NAN);
    } else {
        mean_kernel_impl(iter);
    }
    result
}
pub fn mean(self_: &Tensor, dtype: Option<ScalarType>) -> Tensor {
    return mean_cpu_gpu_(self_, &[], false, dtype);
}

pub fn mean_cpu_gpu_(
    self_: &Tensor,
    dim: &[usize],
    keepdim: bool,
    dtype: Option<ScalarType>,
) -> Tensor {
    let mut result = Tensor::default();
    mean_out_cpu_gpu(&mut result, self_, dim, keepdim, dtype);
    result
}

pub fn sum(self_: &Tensor, dtype: Option<ScalarType>) -> Tensor {
    return sum_dim_int_list(self_, vec![], false, dtype);
}

pub fn sum_dim_int_list(
    self_: &Tensor,
    dim: Vec<usize>,
    keep_dim: bool,
    dtype: Option<ScalarType>,
) -> Tensor {
    let mut result = Tensor::default();
    sum_out(&mut result, self_, dim, keep_dim, dtype);
    result
}

fn get_dtype(
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

fn sum_out<'result>(
    result: &'result mut Tensor,
    self_: &Tensor,
    dim: Vec<usize>,
    keep_dim: bool,
    opt_dtype: Option<ScalarType>,
) -> &'result mut Tensor {
    let dtype = get_dtype(result, self_, opt_dtype, true);
    let iter = make_reduction("sum", result, self_, dim.as_slice(), keep_dim, dtype, dtype);
    if iter.numel() == 0 {
        result.zero_();
    } else {
        let _numel = iter.numel();

        let dtype = iter.dtype();
        match dtype {
            ScalarType::Float => do_reduction::<f32>(&iter, result, self_),
            _ => panic!(),
        };

        sum_kernel_impl(&iter);
    }
    return result;
}

fn do_reduction<T: std::fmt::Debug + num::Zero + std::ops::AddAssign + Copy>(
    iter: &TensorIterator,
    result: &Tensor,
    self_: &Tensor,
) {
    let pt = self_.data_ptr();
    let numel = iter.numel();
    let mut output = T::zero();
    let ptr_ = pt.as_ptr() as *mut T;
    for i in 0..numel {
        unsafe { output += *ptr_.offset(i as isize) };
    }
    unsafe {
        let result_ptr = result.data_ptr().as_ptr() as *mut T;
        *result_ptr = output;
    }
}
