use crate::{
    aten::GRAIN_SIZE,
    c10::{MemoryFormat, ScalarType, TensorOptions},
    tensor::{Tensor, _log_softmax, maybe_wrap_dim},
    AT_DISPATCH_FLOATING_TYPES_AND2, AT_PRIVATE_CASE_TYPE,
};

use super::{cpu::log_softmax_lastdim_kernel_impl, empty_like};

fn host_softmax<T: Ord>(output: Tensor, input: &Tensor, dim: usize) {
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
    let input_data_base = input.data_ptr_casted::<T>();
    let output_data_base = output.data_ptr_casted::<T>();
    let grain_size = (GRAIN_SIZE / dim_size).max(1);

    let closure = |begin: usize, end: usize| {
        for i in begin..end {
            let outer_idx = i / inner_size;
            let inner_idx = i % inner_size;
            let input_data =
                unsafe { input_data_base.offset((outer_idx * outer_stride + inner_idx) as isize) };
            let output_data =
                unsafe { output_data_base.offset((outer_idx * outer_stride + inner_idx) as isize) };
            let max_input = unsafe { *input_data.offset(0) };
            for d in 1..dim_size {
                max_input = max_input.max(unsafe { *input_data.offset((d * dim_stride) as isize) });
            }
        }
    };
    /*
      for (let i = begin; i < end; i++) {
        acc_type<T, false> tmpsum = 0;
        for (let d = 0; d < dim_size; d++) {
          T z = std::exp(input_data[d * dim_stride] - max_input);
          if (!LogSoftMax) {
            output_data[d * dim_stride] = z;
          }
          tmpsum += z;
        }

        if (LogSoftMax)
          tmpsum = std::log(tmpsum);
        else
          tmpsum = 1 / tmpsum;

        for (let d = 0; d < dim_size; d++)
          if (LogSoftMax)
            output_data[d * dim_stride] =
                input_data[d * dim_stride] - max_input - tmpsum;
          else
            output_data[d * dim_stride] *= tmpsum;
      }
    */
}
/*   parallel_for(
        0, outer_size * inner_size, grain_size,
        );
  }
*/

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
    let input = input_.contiguous();
    let output = empty_like(&input, TensorOptions::default(), MemoryFormat::Contiguous);
    let dim = maybe_wrap_dim(dim_, input.dim());

    if input.numel() == 0 {
        return output;
    }
    if input.dim() == 0 {
        input = input.view(1);
    }
    assert!(
        dim >= 0 && dim < input.dim(),
        "dim must be non-negative and less than input dimensions"
    );
    if input.ndimension() > 0 && dim == input.ndimension() - 1 {
        log_softmax_lastdim_kernel_impl(output, input);
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2!(input.scalar_type(), "log_softmax", || {
            host_softmax::<SCALART>(output, input, dim, true)
        });
    }
    return output;
}
