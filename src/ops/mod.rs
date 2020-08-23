mod graphroot;
mod ops;
use crate::tensor::*;
pub use graphroot::*;
pub use ops::*;

pub trait NodeTrait {
    fn call(&mut self, input: Vec<Tensor>) -> Vec<Tensor>;
    fn set_next_edges(&mut self, edges: Vec<Edge>);
    fn add_input_metadata(&mut self, tensor: &Tensor) -> usize;
    fn next_edges(&self) -> Option<&EdgeList>;
    fn next_edge(&self, i: usize) -> Option<Edge>;
    fn num_inputs(&self) -> usize;
    fn num_outputs(&self) -> usize;
}

pub struct Node {
    pub _impl: Box<dyn NodeTrait>,
}

impl Node {
    pub fn new<T>(node_type: T) -> Self
    where
        T: NodeTrait + 'static,
    {
        let t: Box<dyn NodeTrait> = Box::new(node_type);
        Self { _impl: t }
    }
}


impl Node {
    pub fn call(&mut self, input: Vec<Tensor>) -> Vec<Tensor> {
        self._impl.call(input)
    }
    pub fn set_next_edges(&mut self, edges: Vec<Edge>) {
        self._impl.set_next_edges(edges)
    }

    pub fn add_input_metadata(&mut self, tensor: &Tensor) -> usize {
        self._impl.add_input_metadata(tensor)
    }

    pub fn next_edges(&self) -> Option<&EdgeList> {
        self._impl.next_edges()
    }

    pub fn next_edge(&self, i: usize) -> Option<Edge> {
        self._impl.next_edge(i)
    }

    pub fn num_inputs(&self) -> usize {
        self._impl.num_inputs()
    }
    
    pub fn num_outputs(&self) -> usize {
        self._impl.num_outputs()
    }
}
