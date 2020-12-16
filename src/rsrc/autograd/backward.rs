use super::*;
use crate::c10::TensorOptions;
use crate::engine::Engine;
use crate::tensor::*;
use crate::util_autograd;

// Todo: view source for this function. also try to use other two arguments.
// https://github.com/pytorch/pytorch/blob/8850fd1952c3983793dcac4022fdc8e3913dad96/torch/csrc/autograd/autograd.cpp#L127
pub fn backward(tensors: &VariableList, grad_tensors: &VariableList, create_graph: bool) {
    let grads = _make_grads(tensors, grad_tensors);
    // println!("Created grads for backpass: {:?}", grads);
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
        let gradient_edge = util_autograd::gradient_edge(output);
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
                // println!("Pushing new Grads to GradList");
                new_grads.push(ones_like(output, TensorOptions::default()))
            }
        }
    }
    new_grads
}
