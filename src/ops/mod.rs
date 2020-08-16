mod graphroot;
mod ops;
use crate::tensor::*;
pub use graphroot::*;
pub use ops::*;
use std::fmt::Debug;

#[derive(Debug)]
pub enum Node {
    GraphRoot(GraphRoot),
    Function(Function),
    AccumulateGrad(AccumulateGrad),
}




impl Node {
    pub fn call(&mut self, input: Vec<Tensor>) -> Vec<Tensor> {
        match self {
            Node::GraphRoot(root) => root.call(input),
            Node::Function(func) => func.call(input),
            Node::AccumulateGrad(acc) => acc.call(input),
        }
    }

    pub fn set_next_edges(&mut self, edges: Vec<Edge>) {
        match self {
            Node::GraphRoot(_) => todo!(),
            Node::Function(f) => f.set_next_edges(edges),
            Node::AccumulateGrad(_) => {}
        }
    }

    pub fn add_input_metadata(&mut self, tensor: &Tensor) -> usize {
        match self {
            Node::GraphRoot(_) => todo!(),
            Node::Function(f) => f.add_input_metadata(tensor),
            Node::AccumulateGrad(_) => todo!(),
        }
    }

    pub fn next_edges(&self) -> Option<&EdgeList> {
        match self {
            Node::GraphRoot(root) => root.next_edges(),
            Node::Function(f) => f.next_edges(),
            Node::AccumulateGrad(_) => None,
        }
    }

    pub fn next_edge(&self, i: usize) -> Option<Edge> {
        // eprintln!("next_edge: {}", i);
        match self {
            Node::GraphRoot(t) => t.next_edge(i),
            Node::Function(f) => f.next_edge(i),
            Node::AccumulateGrad(_) => todo!(),
        }
    }

    pub fn num_inputs(&self) -> usize {
        match self {
            Node::GraphRoot(_) => todo!(),
            Node::Function(f) => f.num_inputs(),
            Node::AccumulateGrad(a) => a.num_inputs(),
        }
    }

    pub fn num_outputs(&self) -> usize {
        match self {
            Node::GraphRoot(_) => todo!(),
            Node::Function(f) => f.num_outputs(),
            Node::AccumulateGrad(_) => todo!(),
        }
    }
}
