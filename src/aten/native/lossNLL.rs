use std::sync::atomic::AtomicPtr;

use crate::{
    aten::{
        core::{One, PlusOne},
        parallel_for,
    },
    tensor::{loss::Reduction, nll_loss_forward, Tensor},
    AT_DISPATCH_FLOATING_TYPES,
};
use num::Float;

use super::{empty, empty_like};

pub fn nll_loss(
    self_: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    reduction: Reduction,
    ignore_index: i64,
) -> Tensor {
    nll_loss_forward(self_, target, weight, reduction, ignore_index).0
}

pub fn nll_loss_forward_cpu(
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
        weight.is_none() || weight.unwrap().numel() == n_classes,
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
fn optional_data<T>(source: Option<&Tensor>) -> Option<AtomicPtr<T>> {
    if let Some(source) = source {
        if source.defined() {
            Some(AtomicPtr::new(source.data_ptr_casted::<T>()))
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
    unsafe { *total_weight_data = T::zero() };

    let weight_contiguous = optional_contiguous(weight);
    let weight_data = optional_data::<T>(weight_contiguous.as_ref());

    if reduction == Reduction::None && n_dims == 2 {
        let batch_size = input.size(0);
        output.resize(&[batch_size], None);

        let input_acc = input.accessor::<T, PlusOne<One>>(2);
        let target_acc = target.accessor::<i64, One>(1);
        let mut output_acc = output.accessor::<T, One>(1);
        parallel_for(0, batch_size, 0, |begin: usize, end: usize| {
            for i in begin..end {
                let cur_target = target_acc[i];
                if cur_target == ignore_index {
                    output_acc[i] = T::zero();
                    continue;
                }

                assert!(
                    cur_target >= 0 && cur_target < n_classes as i64,
                    "Target {:?} is out of bounds.",
                    cur_target
                );

                let cur_weight = if let Some(data) = weight_data.as_ref() {
                    unsafe {
                        let data = data
                            .load(std::sync::atomic::Ordering::Relaxed)
                            .add(cur_target as usize);
                        data.read()
                    }
                } else {
                    T::one()
                };

                output_acc[i] = input_acc.index(i)[cur_target as usize] * cur_weight;
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
        let cur_target = unsafe { target_data.read() };
        if cur_target != ignore_index {
            assert!(
                cur_target >= 0 && cur_target < n_classes as i64,
                "Target {:?} is out of bounds.",
                cur_target
            );
            total_weight_val = if let Some(data) = weight_data {
                unsafe {
                    let data = data
                        .load(std::sync::atomic::Ordering::Relaxed)
                        .add(cur_target as usize);
                    data.read()
                }
            } else {
                T::one()
            };
            output_val = -unsafe { input_data.add(cur_target as usize).read() } * total_weight_val;
        }
    } else if input.dim() == 2 {
        let batch_size = input.size(0);
        assert!(target.size(0) == batch_size);
        let n_target = input.size(1);

        for i in 0..batch_size {
            let cur_target = unsafe { target_data.add(i).read() };
            if cur_target != ignore_index {
                assert!(
                    cur_target >= 0 && cur_target < n_classes as i64,
                    "Target {:?} is out of bounds.",
                    cur_target
                );

                let cur_weight = if let Some(data) = weight_data.as_ref() {
                    unsafe {
                        let data = data
                            .load(std::sync::atomic::Ordering::Relaxed)
                            .add(cur_target as usize);
                        data.read()
                    }
                } else {
                    T::one()
                };
                total_weight_val = total_weight_val + cur_weight;
                output_val = output_val
                    - unsafe { input_data.add(i * n_target + cur_target as usize).read() }
                        * cur_weight;
            }
        }
    }

    if reduction == Reduction::Mean && (total_weight_val != T::zero() || input.numel() == 0) {
        // allow NaN result for total_weight_val == 0 case, see #15870
        output_val = output_val / total_weight_val;
    }

    // write result to output tensors
    unsafe {
        *output.data_ptr_casted::<T>() = output_val;
        *total_weight_data = total_weight_val;
    }
}
fn nll_loss_backward_out_frame<T: Float>(
    grad_input: &Tensor,
    grad_output: &Tensor,
    input: &Tensor,
    target: &Tensor,
    weight: &Tensor,
    reduction: Reduction,
    ignore_index: i64,
    total_weight: &Tensor,
) {
    let n_dims = input.dim();
    let n_classes = input.size(-1);
    let target_acc = target.accessor::<i64, One>(1);
    let weight_contiguous = optional_contiguous(Some(weight));
    let weight_data = optional_data::<T>(weight_contiguous.as_ref());
    if reduction == Reduction::None && n_dims == 2 {
        let batch_size = input.size(0);
        let grad_input_acc = grad_input.accessor::<T, PlusOne<One>>(2);
        let grad_output_acc = grad_output.accessor::<T, One>(1);
        parallel_for(0, batch_size, 0, |start: usize, end: usize| {
            for i in start..end {
                let cur_target = target_acc[i];
                if cur_target == ignore_index {
                    continue;
                }
                let w = if let Some(w) = weight_data.as_ref() {
                    unsafe {
                        let ptr = w
                            .load(std::sync::atomic::Ordering::Relaxed)
                            .offset(cur_target as isize);
                        ptr.read()
                    }
                } else {
                    T::one()
                };
                let mut t = grad_input_acc.index(i);
                t[cur_target as usize] = -w * grad_output_acc[i];
            }
        })
    }
    let total_weight_value = unsafe { total_weight.data_ptr_casted::<T>().read() };
    if total_weight_value <= T::zero() {
        return;
    }
    assert!(
        grad_output.dim() <= 1 && grad_output.numel() == 1,
        "Expected a single element grad_output tensor, but got: {:?}",
        grad_output.sizes()
    );
    let grad_output_value = unsafe { grad_output.data_ptr_casted::<T>().read() };
    if input.dim() == 1 {
        let mut grad_input_acc = grad_input.accessor::<T, One>(1);
        let cur_target = target_acc[0];
        if cur_target != ignore_index {
            assert!(
                cur_target >= 0 && cur_target < n_classes as i64,
                "Target {} is out of bounds.",
                cur_target
            );
        }
        let item = match weight_data.as_ref() {
            Some(w) => {
                if reduction != Reduction::Mean {
                    unsafe {
                        -w.load(std::sync::atomic::Ordering::Relaxed)
                            .offset(cur_target as isize)
                            .read()
                    }
                } else {
                    -T::one()
                }
            }
            None => -T::one(),
        };
        grad_input_acc[cur_target as usize] = item * grad_output_value;
    } else if input.dim() == 2 {
        let grad_input_acc = grad_input.accessor::<T, PlusOne<One>>(2);

        let batch_size = input.size(0);
        assert!(target.size(0) == batch_size);

        for i in 0..batch_size {
            let cur_target = target_acc[i];

            if cur_target != ignore_index {
                assert!(
                    cur_target >= 0 && cur_target < n_classes as i64,
                    "Target {} is out of bounds.",
                    cur_target
                );

                let w = if let Some(w) = weight_data.as_ref() {
                    unsafe {
                        let val = w
                            .load(std::sync::atomic::Ordering::Relaxed)
                            .offset(cur_target as isize);
                        val.read()
                    }
                } else {
                    T::one()
                };

                grad_input_acc.index(i)[cur_target as usize] = -w * grad_output_value;
                if reduction == Reduction::Mean {
                    grad_input_acc.index(i)[cur_target as usize] =
                        grad_input_acc.index(i)[cur_target as usize] / total_weight_value;
                }
            }
        }
    }
}

fn nll_loss_backward_out_cpu_template(
    grad_input: &mut Tensor,
    grad_output: &Tensor,
    input: &Tensor,
    target: &Tensor,
    weight: &Tensor,
    reduction: Reduction,
    ignore_index: i64,
    total_weight: &Tensor,
) {
    assert!(
        input.dim() > 0 && input.dim() <= 2,
        "input tensor should be 1D or 2D"
    );

    assert!(
        target.dim() == 1,
        "1D target tensor expected, multi-target not supported"
    );
    assert!(
        input.size(0) == target.size(0),
        "size mismatch (got input: {:?}, target: {:?}",
        input.sizes(),
        target.sizes(),
    );
    assert!(
        total_weight.numel() == 1,
        "expected total_weight to be a  single element tensor, got: {:?}, ({} elements)",
        total_weight.sizes(),
        total_weight.numel(),
    );

    grad_input.resize_as_(input);
    grad_input.zero_();

    assert!(grad_input.is_contiguous(), "grad_input must be contiguous");
    assert!(
        !weight.defined() || weight.numel() == input.size(-1),
        "weight tensor should be defined either for all or no classes"
    );
    AT_DISPATCH_FLOATING_TYPES!(input.scalar_type(), "nll_loss_backward_out_frame", || {
        nll_loss_backward_out_frame::<SCALART>(
            grad_input,
            grad_output,
            input,
            target,
            weight,
            reduction,
            ignore_index,
            total_weight,
        );
    });
}

pub fn nll_loss_backward_out_cpu<'a>(
    grad_input: &'a mut Tensor,
    grad_output: &Tensor,
    self_: &Tensor,
    target: &Tensor,
    weight: &Tensor,
    reduction: Reduction,
    ignore_index: i64,
    total_weight: &Tensor,
) -> &'a Tensor {
    nll_loss_backward_out_cpu_template(
        grad_input,
        grad_output,
        self_,
        target,
        weight,
        reduction,
        ignore_index,
        total_weight,
    );
    grad_input
}

pub fn nll_loss_backward_cpu(
    grad_output: &Tensor,
    self_: &Tensor,
    target: &Tensor,
    weight: &Tensor,
    reduction: Reduction,
    ignore_index: i64,
    total_weight: &Tensor,
) -> Tensor {
    let mut grad_input = empty_like(self_, self_.options(), None);
    nll_loss_backward_out_cpu(
        &mut grad_input,
        grad_output,
        self_,
        target,
        weight,
        reduction,
        ignore_index,
        total_weight,
    );
    grad_input
}
