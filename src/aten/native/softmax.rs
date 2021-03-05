use std::sync::atomic::{AtomicPtr, Ordering};

use num::Float;

use crate::{
    aten::{parallel_for, GRAIN_SIZE},
    c10::{MemoryFormat, ScalarType, TensorOptions},
    tensor::{Tensor, _log_softmax, maybe_wrap_dim},
    AT_DISPATCH_FLOATING_TYPES,
};

use super::{
    cpu::{log_softmax_backward_lastdim_kernel_impl, log_softmax_lastdim_kernel_impl},
    empty_like,
};

fn host_softmax<T: Float + Send>(output: &Tensor, input: &Tensor, dim: usize, log_softmax: bool) {
    let dim_size = input.size(dim as i64);
    let outer_size: usize = input.sizes().iter().take(dim).product();
    let inner_size: usize = input.sizes()[(dim + 1)..input.ndimension()]
        .iter()
        .product();
    let dim_stride = inner_size;
    let outer_stride = dim_size * dim_stride;
    let input_data_base = AtomicPtr::new(input.data_ptr_casted::<T>());
    let output_data_base = AtomicPtr::new(output.data_ptr_casted::<T>());
    let grain_size = (GRAIN_SIZE / dim_size).max(1);

    let closure = |begin: usize, end: usize| {
        for i in begin..end {
            let outer_idx = i / inner_size;
            let inner_idx = i % inner_size;
            let input_data = unsafe {
                let input_data = input_data_base
                    .load(Ordering::Relaxed)
                    .add(outer_idx * outer_stride + inner_idx);
                let input_data = std::slice::from_raw_parts(input_data, outer_stride);
                input_data
            };
            let output_data = unsafe {
                let output_data = output_data_base
                    .load(Ordering::Relaxed)
                    .add(outer_idx * outer_stride + inner_idx);
                let output_data = std::slice::from_raw_parts_mut(output_data, outer_stride);
                output_data
            };

            let mut max_input = input_data[0];
            for d in 1..dim_size {
                max_input = if max_input > input_data[(d * dim_stride)] {
                    max_input
                } else {
                    input_data[d * dim_stride]
                };
            }
            let mut tmpsum = T::zero();
            for d in 0..dim_size {
                let mut z: T = input_data[d * dim_stride] - max_input;
                z = z.exp();
                if !log_softmax {
                    output_data[d * dim_stride] = z;
                }
                tmpsum = tmpsum + z;
            }
            if log_softmax {
                tmpsum = tmpsum.ln();
            } else {
                tmpsum = T::one() / tmpsum;
            }
            for d in 0..dim_size {
                if log_softmax {
                    output_data[d * dim_stride] = input_data[d * dim_stride] - max_input - tmpsum;
                } else {
                    output_data[d * dim_stride] = output_data[d * dim_stride] * tmpsum;
                }
            }
        }
    };
    parallel_for(0, outer_size * inner_size, grain_size, closure);
}

pub fn log_softmax(input_: &Tensor, dim_: i64, dtype: Option<ScalarType>) -> Tensor {
    let result = || -> Tensor {
        let converted = if let Some(d) = dtype {
            input_.to_dtype(d)
        } else {
            input_.clone()
        };
        return _log_softmax(&converted, dim_, false);
    }();
    return result;
}

fn host_softmax_backward<T: Float + Send>(
    grad_input: &Tensor,
    grad: &Tensor,
    output: &Tensor,
    dim: usize,
    log_softmax: bool,
) {
    let dim_size = grad.size(dim as i64);
    let outer_size: usize = grad.sizes().iter().take(dim).product();
    let inner_size = grad.sizes()[(dim + 1)..grad.ndimension()].iter().product();
    let dim_stride = inner_size;
    let outer_stride = dim_size * dim_stride;
    let grad_input_data_base = AtomicPtr::new(grad_input.data_ptr_casted::<T>());
    let grad_output_data_base = AtomicPtr::new(grad.data_ptr_casted::<T>());
    let output_data_base = AtomicPtr::new(output.data_ptr_casted::<T>());
    let grain_size = (GRAIN_SIZE / dim_size).min(1);

    let closure = |begin: usize, end: usize| {
        for i in begin..end {
            let outer_idx = i / inner_size;
            let inner_idx = i % inner_size;
            let grad_input_data = unsafe {
                let grad_input_data = grad_input_data_base
                    .load(Ordering::Relaxed)
                    .add(outer_idx * outer_stride + inner_idx);
                std::slice::from_raw_parts_mut(grad_input_data, outer_stride)
            };
            let grad_output_data = unsafe {
                let grad_output_data = grad_output_data_base
                    .load(Ordering::Relaxed)
                    .add(outer_idx * outer_stride + inner_idx);
                std::slice::from_raw_parts(grad_output_data, outer_stride)
            };
            let output_data = unsafe {
                let output_data = output_data_base
                    .load(Ordering::Relaxed)
                    .add(outer_idx * outer_stride + inner_idx);
                std::slice::from_raw_parts(output_data, outer_stride)
            };

            let mut sum = T::zero();
            for d in (0..dim_size).step_by(dim_stride) {
                if log_softmax {
                    sum = sum + grad_output_data[d];
                } else {
                    sum = sum + grad_output_data[d] * output_data[d];
                }
            }

            for d in (0..dim_size).step_by(dim_stride) {
                if log_softmax {
                    grad_input_data[d] =
                        grad_output_data[d * dim_stride] - output_data[d].exp() * sum;
                } else {
                    grad_input_data[d] = output_data[d] * grad_output_data[d] - sum;
                }
            }
        }
    };
    parallel_for(0, outer_size * inner_size, grain_size, closure);
}

pub fn log_softmax_cpu(input_: &Tensor, dim_: i64, half_to_float: bool) -> Tensor {
    assert!(
        !half_to_float,
        "softmax with half to float conversion is not supported on Cpu"
    );
    let mut input = input_.contiguous();
    let output = empty_like(&input, TensorOptions::default(), MemoryFormat::Contiguous);
    let dim = maybe_wrap_dim(dim_, input.dim(), false);

    if input.numel() == 0 {
        return output;
    }
    if input.dim() == 0 {
        input = input.view(&[1]);
    }
    assert!(
        dim < input.ndimension(),
        "dim must be non-negative and less than input dimensions"
    );
    if input.ndimension() > 0 && dim == input.ndimension() - 1 {
        log_softmax_lastdim_kernel_impl(&output, &input);
    } else {
        run_dispatch_log_softmax_forward(&input, &output, dim);
    }
    return output;
}

pub fn log_softmax_backward_cpu(grad_: &Tensor, output_: &Tensor, dim_: i64) -> Tensor {
    let dim = maybe_wrap_dim(dim_, grad_.dim(), false);
    let mut grad = grad_.contiguous();
    let mut output = output_.contiguous();
    let grad_input = empty_like(&grad, TensorOptions::default(), MemoryFormat::Contiguous);
    if output.numel() == 0 {
        return grad_input;
    }

    if grad.dim() == 0 {
        grad = grad.view(&[1]);
    }
    if output.dim() == 0 {
        output = output.view(&[1]);
    }
    assert!(
        dim < grad.dim() as usize,
        "dim must be non-negative and less than input dimensions"
    );
    if grad.ndimension() > 0 && dim == grad.ndimension() - 1 {
        log_softmax_backward_lastdim_kernel_impl(&grad_input, &grad, &output);
    } else {
        run_dispatch_log_softmax_backward(&grad_input, &grad, &output, dim);
    }
    return grad_input;
}

fn run_dispatch_log_softmax_forward(input: &Tensor, output: &Tensor, dim: usize) {
    AT_DISPATCH_FLOATING_TYPES!(input.scalar_type(), "log_softmax", || {
        host_softmax::<Scalart>(output, input, dim, true)
    });
}

fn run_dispatch_log_softmax_backward(
    grad_input: &Tensor,
    grad: &Tensor,
    output: &Tensor,
    dim: usize,
) {
    AT_DISPATCH_FLOATING_TYPES!(grad.scalar_type(), "log_softmax_backward", || {
        host_softmax_backward::<Scalart>(grad_input, grad, output, dim, true)
    });
}
