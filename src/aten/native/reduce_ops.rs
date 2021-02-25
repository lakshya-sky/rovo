use c10::isFloatingType;

use crate::{
    c10::{self, ScalarType},
    tensor::{maybe_wrap_dim, Tensor},
};

use super::{
    super::native,
    cpu::{argmax_kernel_impl, mean_kernel_impl, sum_kernel_impl},
    reduce_ops_utils::{get_dtype, make_reduction},
};

pub fn argmax_out<'result>(
    result: &'result mut Tensor,
    self_: &Tensor,
    dim: Option<i64>,
    mut keepdim: bool,
) -> &'result Tensor {
    assert!(self_.numel() > 0, "cannot perform reduction function argmax on a tensor with no elements because the operation does not have an identity");
    let in_: Tensor;
    let wrap_dim;
    if let Some(dim) = dim {
        let sizes = self_.sizes();
        wrap_dim = maybe_wrap_dim(dim, self_.dim(), true);
        if sizes[wrap_dim] == 1 {
            if keepdim {
                result.move_tensor(native::zeros(
                    sizes,
                    self_.options().set_dtype_(ScalarType::Long),
                ))
            } else {
                let mut sizes_vec = sizes.to_vec();
                sizes_vec.remove(wrap_dim);
                result.move_tensor(native::zeros(
                    sizes_vec.as_slice(),
                    self_.options().set_dtype_(ScalarType::Long),
                ))
            }
            return result;
        }
        in_ = self_.clone();
    } else {
        //in_ = self_.reshape([-1]);
        in_ = native::empty(&[0], self_.options(), None);
        keepdim = false;
        wrap_dim = 0;
    }
    let iter = make_reduction(
        "argmax",
        result,
        &in_,
        &[wrap_dim],
        keepdim,
        self_.scalar_type(),
        ScalarType::Long,
    );
    argmax_kernel_impl(&iter);
    result
}

pub fn argmax(self_: &Tensor, dim: Option<i64>, keepdims: bool) -> Tensor {
    let mut result = native::empty(&[0], self_.options().set_dtype_(ScalarType::Long), None);
    argmax_out(&mut result, self_, dim, keepdims);
    result
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
        mean_kernel_impl(&iter);
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
        sum_kernel_impl(&iter);
    }
    return result;
}
