use super::{tensor_iterator, Tensor};

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
    let mut loss = Tensor::empty_like(input);
    let iter = tensor_iterator::TensorIteratorConfig::default()
        .add_output(&loss)
        .add_input(input)
        .add_input(target)
        .build();

    iter.for_each(|i: &f64, t: &f64| {
        let result = (t - 1.0) * (1.0 - i).ln().max(-100.0) - (t * i.ln().max(-100.0));
        result
    });

    if let Some(weight) = weight {
        loss.mul_(weight);
    }

    if reduction != Reduction::None {
        let loss_reduced = apply_loss_reduction(&loss, reduction);
        // Todo: instead of the following line Pytorch uses resize_as_ and then copy_
        // see: Loss.cpp: binary_cross_entropy_out_cpu
        loss.get_tensor_impl().data = loss_reduced.get_tensor_impl().data.clone();
    }
    loss
}

pub fn binary_cross_entropy_backward(
    grad: &Tensor,
    input: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    reduction: Reduction,
) -> Tensor {
    let mut grad_input = Tensor::empty_like(input);
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
    const EPSILON: f64 = 1e-12;
    let mut grad_iter = grad.get_tensor_impl().data.iter();
    let mut input_iter = input.get_tensor_impl().data.iter();
    let mut target_iter = target.get_tensor_impl().data.iter();
    let op = |g: &f64, i: &f64, t: &f64| {
        let result = g * (i - t) / (((1.0 - i) * i).max(EPSILON));
        return result;
    };
    let columns = grad.sizes()[1];
    let mut row;
    let mut col;
    let mut idx = 0usize;
    {
        let data = &mut grad_input.get_tensor_impl().data;
        loop {
            match (grad_iter.next(), input_iter.next(), target_iter.next()) {
                (Some(g), Some(i), Some(t)) => {
                    let result = op(g, i, t);
                    row = idx / columns;
                    col = idx % columns;
                    data[[row, col]] = result;
                }
                _ => break,
            }
            idx += 1;
        }
    }

    if let Some(weight) = weight {
        grad_input.mul_(weight);
    }

    match reduction {
        Reduction::Mean => grad_input.div_(&Tensor::from_scalar(
            grad_input.shape(),
            input.numel() as f64,
            false,
        )),
        _ => (),
    }
}

#[cfg(test)]
mod test {
    use crate::tensor::{Reduction, Tensor};
    #[test]
    fn bce_loss_test() {
        let input = Tensor::from_scalar(&[2, 2], 2.0, false);
        let target = Tensor::from_scalar(&[2, 2], 3.0, false);
        let result = super::binary_cross_entropy(&input, &target, None, Reduction::Mean);
        println!("BCE Result: {:?}", result);
    }

    #[test]
    fn bce_loss_backward_test() {
        let input = Tensor::from_scalar(&[2, 2], 2.0, false);
        let target = Tensor::from_scalar(&[2, 2], 3.0, false);
        let grad = Tensor::from_scalar(&[2, 2], 1.0, false);
        let result =
            super::binary_cross_entropy_backward(&grad, &input, &target, None, Reduction::Mean);
        println!("BCE Result: {:?}", result);
    }
}