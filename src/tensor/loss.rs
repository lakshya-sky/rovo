use aten::native::cpu_kernel;

use super::{Tensor, TensorIterator, TensorIteratorConfig};
use crate::{aten, autograd, Closure, AT_DISPATCH_FLOATING_TYPES};

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum Reduction {
    None,
    Mean,
    Sum,
}
fn apply_loss_reduction(unreduced: &Tensor, reduction: Reduction) -> Tensor {
    match reduction {
        Reduction::None => unreduced.clone(),
        Reduction::Mean => unreduced.mean(),
        Reduction::Sum => unreduced.sum(),
    }
}

pub fn binary_cross_entropy(
    input: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    reduction: Reduction,
) -> Tensor {
    let mut loss = autograd::empty_like(input, None, None);
    binary_cross_entropy_out_cpu(&mut loss, input, target, weight, reduction);
    loss
}

pub fn binary_cross_entropy_out_cpu<'r>(
    loss: &'r mut Tensor,
    input: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    reduction: Reduction,
) -> &'r Tensor {
    let loss_squeezed = aten::native::squeeze(loss);

    let mut iter = TensorIteratorConfig::default()
        .add_output(&loss_squeezed)
        .add_input(&aten::native::squeeze(input))
        .add_input(&aten::native::squeeze(target))
        .build();
    binary_cross_entropy_kernel(&mut iter);

    if let Some(weight) = weight {
        loss.mul_(weight);
    }

    if reduction != Reduction::None {
        let loss_reduced = apply_loss_reduction(loss, reduction);
        loss.resize_as_(&loss_reduced).copy(&loss_reduced, None);
    }
    loss
}

pub fn binary_cross_entropy_kernel(iter: &mut TensorIterator) {
    AT_DISPATCH_FLOATING_TYPES!(iter.dtype(), "binary_cross_entropy", || {
        cpu_kernel(
            iter,
            Closure::new(|args: [Scalart; 2]| -> Scalart {
                let i = args[0];
                let t = args[1];
                assert!(
                    (i >= 0.0) && (i <= 1.0),
                    "all elements of input should be between 0 and 1"
                );
                (t - 1.0) * (1.0 - i).ln().max(-100.0) - (t * i.ln().max(-100.0))
            }),
        )
    });
}

pub fn binary_cross_entropy_backward(
    grad: &Tensor,
    input: &Tensor,
    target: &Tensor,
    weight: &Tensor,
    reduction: Reduction,
) -> Tensor {
    let mut grad_input = autograd::empty_like(input, None, None);
    binary_cross_entropy_backward_out(&mut grad_input, grad, input, target, weight, reduction);
    grad_input
}

pub fn binary_cross_entropy_backward_out<'r>(
    grad_input: &'r mut Tensor,
    grad: &Tensor,
    input: &Tensor,
    target: &Tensor,
    weight: &Tensor,
    reduction: Reduction,
) -> &'r Tensor {
    let grad_input_squeezed = aten::native::squeeze(grad_input);
    let mut iter = TensorIteratorConfig::default()
        .add_output(&grad_input_squeezed)
        .add_input(&aten::native::squeeze(grad))
        .add_input(&aten::native::squeeze(input))
        .add_input(&aten::native::squeeze(target))
        .build();

    binary_cross_entropy_backward_kernel(&mut iter);

    if weight.defined() {
        grad_input.mul_(weight);
    }

    match reduction {
        Reduction::Mean => grad_input.div_scalar(input.numel()),
        _ => (),
    }
    grad_input
}

pub fn binary_cross_entropy_backward_kernel(iter: &mut TensorIterator) {
    const EPSILON: f64 = 1e-12;
    AT_DISPATCH_FLOATING_TYPES!(iter.dtype(), "binary_cross_entropy_backward", || {
        cpu_kernel(
            iter,
            Closure::new(|args: [Scalart; 3]| -> Scalart {
                let g = args[0];
                let i = args[1];
                let t = args[2];
                g * (i - t) / (((1.0 - i) * i).max(EPSILON as Scalart))
            }),
        )
    });
}
#[cfg(test)]
mod test {
    use crate::{autograd, tensor};
    use crate::{
        c10::TensorOptions,
        tensor::{loss::Reduction, sigmoid},
    };
    #[test]
    fn bce_loss_test() {
        crate::init_rovo();
        let input = autograd::full(&[2, 3], 1.5, TensorOptions::with_requires_grad());
        let target = autograd::full(&[2, 3], 1.0, TensorOptions::with_requires_grad());
        let result = super::binary_cross_entropy(&sigmoid(&input), &target, None, Reduction::Mean);
        println!("BCE Result: {:?}", result);
    }

    #[test]
    fn bce_loss_backward_test() {
        crate::init_rovo();
        let input = autograd::full(&[2, 3], 1.5, TensorOptions::with_requires_grad());
        let target = autograd::full(&[2, 3], 1.0, None);
        let result = tensor::binary_cross_entropy(&sigmoid(&input), &target, None, Reduction::Mean);
        println!("BCE Result: {:?}", result);
        crate::autograd::backward::backward(&vec![result], &vec![], false);
        println!("Input Grad: {:?}", input.grad());

        //Expected :
        // BCE Result: Tensor: [0.20141332]        size: []
        // Input Grad: Some(Tensor: [-0.03040426, -0.03040426, -0.03040426, -0.03040426, -0.03040426, -0.03040426] size: [2, 3])
    }
}
