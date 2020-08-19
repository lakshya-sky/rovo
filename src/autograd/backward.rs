use crate::engine::Engine;
use crate::tensor::*;
use crate::util;

// Todo: view source for this function. also try to use other two arguments.
// https://github.com/pytorch/pytorch/blob/8850fd1952c3983793dcac4022fdc8e3913dad96/torch/csrc/autograd/autograd.cpp#L127
pub fn backward(tensors: &VariableList, grad_tensors: &VariableList, create_graph: bool) {
    let grads = _make_grads(tensors, grad_tensors);
    run_backward(tensors, grads, create_graph, &mut vec![])
}

fn run_backward(
    outputs: &VariableList,
    grad_outputs: VariableList,
    create_graph: bool,
    _inputs: &mut VariableList,
) {
    let num_tensors = outputs.len();
    let mut roots: EdgeList = vec![];
    roots.reserve(num_tensors);
    for output in outputs {
        let gradient_edge = util::gradient_edge(output);
        roots.push(gradient_edge);
    }
    let mut output_edges: EdgeList = vec![];
    Engine::get_default_engine().execute(roots, grad_outputs, create_graph, &mut output_edges);
}

fn _make_grads(outputs: &VariableList, grad_outputs: &VariableList) -> VariableList {
    let num_tensors = outputs.len();
    let _num_gradients = grad_outputs.len();
    let mut new_grads: VariableList = vec![];
    new_grads.reserve(num_tensors);
    if grad_outputs.is_empty() {
        for output in outputs {
            if output.requires_grad() {
                new_grads.push(Tensor::ones_like(output))
            }
        }
    }
    new_grads
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_backward_add() {
        let t = Tensor::from_scalar(&[2, 2], 2.0, true);
        let x = Tensor::from_scalar(&[2, 2], 1.0, true);
        let res: Tensor = &t + &x;

        backward(&vec![res], &vec![], false);
        // t.grad().unwrap();
        println!("{}", t.grad().unwrap().as_ref()._impl.borrow().data);
        // println!("{}", x.grad().unwrap().as_ref()._impl.borrow().data);
    }
    #[test]
    fn test_backward_mul() {
        let t = Tensor::from_scalar(&[2, 2], 7.0, true);
        let x = Tensor::from_scalar(&[2, 2], 3.0, true);
        let res: Tensor = &t * &x;

        backward(&vec![res], &vec![], false);
        // t.grad().unwrap();
        println!("{}", t.grad().unwrap().as_ref()._impl.borrow().data);
        println!("{}", x.grad().unwrap().as_ref()._impl.borrow().data);
    }

    #[test]
    fn test_backward_mul_add() {
        let t = Tensor::from_scalar(&[2, 2], 7.0, true);
        let x = Tensor::from_scalar(&[2, 2], 3.0, true);
        let add: Tensor = &t + &x;
        let mul = &x * &add;
        backward(&vec![mul], &vec![], false);
        // t.grad().unwrap();
        println!("{}", t.grad().unwrap().as_ref()._impl.borrow().data);
        println!("{}", x.grad().unwrap().as_ref()._impl.borrow().data);
    }
}
