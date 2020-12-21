use std::sync::atomic::AtomicPtr;

use num::Float;

use crate::{
    aten::{parallel_for, GRAIN_SIZE},
    c10::{MemoryFormat, ScalarType, TensorOptions},
    tensor::{Tensor, _log_softmax, maybe_wrap_dim},
    AT_DISPATCH_FLOATING_TYPES, AT_PRIVATE_CASE_TYPE,
};

use super::{cpu::log_softmax_lastdim_kernel_impl, empty_like};

fn host_softmax<T: Float + Send>(output: &Tensor, input: &Tensor, dim: usize, log_softmax: bool) {
    let mut outer_size = 1;
    let dim_size = input.size(dim);
    let mut inner_size = 1;
    for i in 0..dim {
        outer_size *= input.size(i);
    }
    for i in dim + 1..(input.dim() as usize) {
        inner_size *= input.size(i);
    }
    let dim_stride = inner_size;
    let outer_stride = dim_size * dim_stride;
    let mut input_data_base = AtomicPtr::new(input.data_ptr_casted::<T>());
    let mut output_data_base = AtomicPtr::new(output.data_ptr_casted::<T>());
    let grain_size = (GRAIN_SIZE / dim_size).max(1);

    let closure = |begin: usize, end: usize| {
        for i in begin..end {
            let outer_idx = i / inner_size;
            let inner_idx = i % inner_size;
            let input_data = unsafe {
                let input_data = input_data_base
                    .get_mut()
                    .offset((outer_idx * outer_stride + inner_idx) as isize);
                let input_data = std::slice::from_raw_parts(input_data, outer_stride);
                input_data
            };
            let output_data = unsafe {
                let output_data = output_data_base
                    .get_mut()
                    .offset((outer_idx * outer_stride + inner_idx) as isize);
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
                let z: T = (input_data[d * dim_stride] - max_input).exp();
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
        let converted = if dtype.is_some() {
            input_.to_dtype(dtype.unwrap())
        } else {
            input_.clone()
        };
        return _log_softmax(&converted, dim_, false);
    }();
    return result;
}

pub fn log_softmax_cpu(input_: &Tensor, dim_: i64, half_to_float: bool) -> Tensor {
    assert!(
        !half_to_float,
        "softmax with half to float conversion is not supported on CPU"
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
        run_dispatch(&input, &output, dim);
    }
    return output;
}
fn run_dispatch(input: &Tensor, output: &Tensor, dim: usize) {
    AT_DISPATCH_FLOATING_TYPES!(input.scalar_type(), "log_softmax", || {
        host_softmax::<SCALART>(output, input, dim, true)
    });
}
