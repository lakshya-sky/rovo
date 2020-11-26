use super::{Tensor, TensorIteratorConfig};
use crate::autograd;

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
    _weight: Option<&Tensor>,
    _reduction: Reduction,
) -> Tensor {
    let mut _loss = autograd::empty_like(input, None, None);
    let _iter = TensorIteratorConfig::default()
        .add_output(&_loss)
        .add_input(input)
        .add_input(target)
        .build();

    // iter.for_each(|i: &f64, t: &f64| {
    //     let result = (t - 1.0) * (1.0 - i).ln().max(-100.0) - (t * i.ln().max(-100.0));
    //     result
    // });

    // if let Some(weight) = weight {
    //     loss.mul_(weight);
    // }

    // if reduction != Reduction::None {
    //     let loss_reduced = apply_loss_reduction(&loss, reduction);
    //     // Todo: instead of the following line Pytorch uses resize_as_ and then copy_
    //     // see: Loss.cpp: binary_cross_entropy_out_cpu
    //     loss.get_tensor_impl().data = loss_reduced.get_tensor_impl().data.clone();
    // }
    // loss
    todo!()
}

pub fn binary_cross_entropy_backward(
    grad: &Tensor,
    input: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    reduction: Reduction,
) -> Tensor {
    let mut grad_input = autograd::empty_like(input, None, None);
    binary_cross_entropy_backward_out(&mut grad_input, grad, input, target, weight, reduction);
    grad_input
}

pub fn binary_cross_entropy_backward_out(
    grad_input: &mut Tensor,
    grad: &Tensor,
    input: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    reduction: Reduction,
) {
    // eprintln!("Grad: {:?}", grad);
    const EPSILON: f64 = 1e-12;

    let _iter = TensorIteratorConfig::default()
        .add_output(grad_input)
        .add_input(grad)
        .add_input(input)
        .add_input(target)
        .build();

    let _op = |g: &f64, i: &f64, t: &f64| {
        let result = g * (i - t) / (((1.0 - i) * i).max(EPSILON));
        return result;
    };
    // iter.for_each_ternary(op);
    // todo!();

    if let Some(weight) = weight {
        grad_input.mul_(weight);
    }

    match reduction {
        Reduction::Mean => grad_input.div_(&autograd::full(
            grad_input.sizes(),
            input.numel() as f32,
            None,
        )),
        _ => (),
    }
}

// #[cfg(test)]
// mod test {
//     use crate::tensor;
//     use crate::tensor::{loss::Reduction, sigmoid, Tensor};
//     #[test]
//     fn bce_loss_test() {
//         let input = Tensor::from_scalar(&[2, 3], 2.0, false);
//         let target = Tensor::from_scalar(&[2, 3], 1.0, false);
//         let result = super::binary_cross_entropy(&sigmoid(&input), &target, None, Reduction::Mean);
//         println!("BCE Result: {:?}", result);
//     }

//     #[test]
//     fn bce_loss_backward_test() {
//         let input = Tensor::from_scalar(&[2, 3], 1.0, true);
//         let target = Tensor::from_scalar(&[2, 3], 1.0, false);
//         let result = tensor::binary_cross_entropy(&sigmoid(&input), &target, None, Reduction::Mean);
//         println!("BCE Result: {:?}", result);
//         crate::autograd::backward::backward(&vec![result], &vec![], false);
//         println!("Input Grad: {:?}", input.grad());
//     }
// }
