use std::slice::Windows;

use num::Float;

use crate::{
    aten::parallel_for,
    c10::{ScalarType, TensorOptions},
    tensor::{loss::Reduction, Tensor},
    AT_DISPATCH_FLOATING_TYPES, AT_PRIVATE_CASE_TYPE,
};

use super::empty;

pub fn nll_loss(
    self_: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    reduction: Reduction,
    ignore_index: i64,
) -> Tensor {
    nll_loss_forward_cpu(self_, target, weight, reduction, ignore_index).0
}

fn nll_loss_forward_cpu(
    self_: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    reduction: Reduction,
    ignore_index: i64,
) -> (Tensor, Tensor) {
    let mut output = empty(&[0], self_.options(), None);
    let mut total_weight = empty(&[0], self_.options(), None);
    nll_loss_forward_out_cpu(
        &mut output,
        &mut total_weight,
        self_,
        target,
        weight,
        reduction,
        ignore_index,
    );
    (output, total_weight)
}

fn nll_loss_forward_out_cpu<'o, 't>(
    output: &'o Tensor,
    total_weight: &'t Tensor,
    self_: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    reduction: Reduction,
    ignore_index: i64,
) -> (&'o Tensor, &'t Tensor) {
    nll_loss_forward_out_cpu_template(
        output,
        total_weight,
        self_,
        target,
        weight,
        reduction,
        ignore_index,
    );
    (output, total_weight)
}

fn nll_loss_forward_out_cpu_template(
    output: &Tensor,
    total_weight: &Tensor,
    input: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    reduction: Reduction,
    ignore_index: i64,
) {
    assert!(
        input.dim() > 0 && input.dim() <= 2,
        "Input tensor should be 1D or 2D"
    );
    assert!(
        target.dim() == 1,
        "1D target tensor expected, multi-target not supported"
    );
    assert!(input.size(0) == target.size(0));
    let n_classes = input.size(-1);
    assert!(
        weight.is_some() || weight.unwrap().numel() == n_classes,
        "weight tensor should be defined either for all {} classes or no classes but got weight tensor of shape: {:?}",
        n_classes,
        weight.unwrap().sizes()
    );
    total_weight.resize(&[], None);
    AT_DISPATCH_FLOATING_TYPES!(input.scalar_type(), "nll_loss_out_frame", || {
        nll_loss_out_frame::<SCALART>(
            output,
            total_weight,
            input,
            target,
            weight,
            reduction,
            ignore_index,
        );
    });
}

// Returns a contiguous tensor if the source tensor
// is defined. Otherwise returns the undefined
// source tensor unmodified.
#[inline(always)]
fn optional_contiguous(source: Option<&Tensor>) -> Option<Tensor> {
    if let Some(source) = source {
        if source.defined() {
            Some(source.contiguous())
        } else {
            None
        }
    } else {
        None
    }
}

// Returns the address of the first element of a tensor
// or nullptr if the tensor is undefined.
#[inline(always)]
fn optional_data<T>(source: Option<&Tensor>) -> Option<*mut T> {
    if let Some(source) = source {
        if source.defined() {
            Some(source.data_ptr_casted::<T>())
        } else {
            None
        }
    } else {
        None
    }
}

fn nll_loss_out_frame<T: Float>(
    output: &Tensor,
    total_weight: &Tensor,
    input: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    reduction: Reduction,
    ignore_index: i64,
) {
    let n_dims = input.dim();
    let n_classes = input.size(-1);
    let total_weight_data = total_weight.data_ptr_casted::<T>();
    unsafe { *total_weight_data = 0 };

    let weight_contiguous = optional_contiguous(weight);
    let weight_data = optional_data::<T>(weight_contiguous.as_ref());

    if reduction == Reduction::None && n_dims == 2 {
        let batch_size = input.size(0);
        output.resize(&[batch_size], None);

        let input_acc = input.accessor::<T, 2>();
        let target_acc = target.accessor::<T, 1>();
        let output_acc = output.accessor::<T, 1>();

        parallel_for(0, batch_size, 0, |begin: usize, end: usize| {
            for i in begin..end {
                let cur_target = target_acc[i];
                if cur_target == ignore_index {
                    output_acc[i] = 0;
                    continue;
                }

                assert!(
                    cur_target >= 0 && cur_target < n_classes,
                    "Target {:?} is out of bounds.",
                    cur_target
                );

                let cur_weight = if let Some(data) = weight_data {
                    unsafe {
                        let data = data.add(cur_target);
                        data.read()
                    }
                } else {
                    T::one()
                };
                output_acc[i] = -input_acc[i][cur_target] * cur_weight;
            }
        });

        return;
    }
    // produce scalar output when reducing or input is 1d
    output.resize(&[], None);

    let input_contiguous = input.contiguous();
    let target_contiguous = target.contiguous();

    let input_data = input_contiguous.data_ptr_casted::<T>();
    let target_data = target_contiguous.data_ptr_casted::<i64>();

    let mut output_val = T::zero();
    let mut total_weight_val = T::zero();

    if input.dim() == 1 {
        let cur_target = target_data[0];
        if cur_target != ignore_index {
            assert!(
                cur_target >= 0 && cur_target < n_classes,
                "Target {:?} is out of bounds.",
                cur_target
            );
            total_weight_val = if let Some(data) = weight_data {
                unsafe {
                    let data = data.add(cur_target);
                    data.read()
                }
            } else {
                T::one()
            };
            output_val = -input_data[cur_target] * total_weight_val;
        }
    } else if input.dim() == 2 {
        let batch_size = input.size(0);
        assert!(target.size(0) == batch_size);
        let n_target = input.size(1);

        for i in 0..batch_size {
            let cur_target = target_data[i];
            if cur_target != ignore_index {
                assert!(
                    cur_target >= 0 && cur_target < n_classes,
                    "Target {:?} is out of bounds.",
                    cur_target
                );

                let cur_weight = if let Some(data) = weight_data {
                    unsafe {
                        let data = data.add(cur_target);
                        data.read()
                    }
                } else {
                    T::one()
                };
                total_weight_val = total_weight_val + cur_weight;
                output_val = output_val - input_data[i * n_target + cur_target] * cur_weight;
            }
        }
    }

    if reduction == Reduction::Mean && (total_weight_val != 0 || input.numel() == 0) {
        // allow NaN result for total_weight_val == 0 case, see #15870
        output_val = output_val / total_weight_val;
    }

    // write result to output tensors
    unsafe {
        *output.data_ptr_casted::<T>() = output_val;
        *total_weight_data = total_weight_val;
    }
}
