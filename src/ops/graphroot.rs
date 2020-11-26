use crate::ops::NodeTrait;
use crate::tensor::*;
pub struct GraphRoot {
    next_edges: Option<EdgeList>,
    outputs: VariableList,
}

impl GraphRoot {
    pub fn new(functions: EdgeList, inputs: VariableList) -> GraphRoot {
        GraphRoot {
            next_edges: Some(functions),
            outputs: inputs,
        }
    }
}

impl NodeTrait for GraphRoot {
    fn call(&mut self, _input: VariableList) -> VariableList {
        // self.outputs
        // let outputs = self.outputs.as_slice().to_vec();
        // outputs
        self.outputs.clone()
    }

    fn set_next_edges(&mut self, _edges: EdgeList) {
        todo!()
    }

    fn add_input_metadata(&mut self, _tensor: &Tensor) -> usize {
        todo!()
    }

    fn next_edges(&self) -> Option<&EdgeList> {
        self.next_edges.as_ref()
    }

    fn next_edge(&self, i: usize) -> Option<Edge> {
        let edges = self.next_edges.as_ref().unwrap();
        let e = edges.get(i).and_then(|e| Some(e.clone()));
        e
    }

    fn num_inputs(&self) -> usize {
        todo!()
    }

    fn num_outputs(&self) -> usize {
        todo!()
    }

    fn input_metadata(&self, _index: usize) -> &super::InputMetaData {
        todo!()
    }

    fn debug_print(&self) -> String {
        "Graph Root".to_string()
    }
}
