mod graph_ops;
mod graphroot;
use crate::tensor::*;
pub use graph_ops::*;
pub use graphroot::*;

pub trait NodeTrait {
    fn call(&mut self, input: Vec<Tensor>) -> Vec<Tensor>;
    fn set_next_edges(&mut self, edges: Vec<Edge>);
    fn add_input_metadata(&mut self, tensor: &Tensor) -> usize;
    fn next_edges(&self) -> Option<&EdgeList>;
    fn next_edge(&self, i: usize) -> Option<Edge>;
    fn num_inputs(&self) -> usize;
    fn num_outputs(&self) -> usize;
    fn input_metadata(&self, index: usize) -> &InputMetaData;
    fn debug_print(&self) -> String;
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

    pub fn input_metadata(&self, index: usize) -> &InputMetaData {
        self._impl.input_metadata(index)
    }
}

impl std::fmt::Debug for dyn NodeTrait {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.debug_print())
    }
}
