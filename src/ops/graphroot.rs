use crate::tensor::*;

#[derive(Debug)]
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

    pub fn call(&mut self, _input: VariableList) -> VariableList {
        // self.outputs
        let outputs = self.outputs.as_slice().to_vec();
        outputs
        // self.outputs.clone()
    }

    pub fn set_next_edges(&mut self, _edges: EdgeList) {
        todo!()
    }

    pub fn add_input_metadata(&mut self, _tensor: &Tensor) -> usize {
        todo!()
    }

    pub fn next_edges(&self) -> Option<&EdgeList> {
        self.next_edges.as_ref()
    }

    pub fn next_edge(&self, i: usize) -> Option<Edge> {
        let edges = self.next_edges.as_ref().unwrap();
        let e = edges.get(i).and_then(|e| Some(e.clone()));
        e
    }

    pub fn num_inputs(&self) -> usize {
        todo!()
    }

    pub fn num_outputs(&self) -> usize {
        todo!()
    }
}
