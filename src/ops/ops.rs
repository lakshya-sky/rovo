use crate::autograd::SavedTensor;
use crate::tensor::*;
use smallvec::*;
use std::rc::Rc;

#[derive(Debug)]
pub struct AccumulateGrad {
    tensor: Tensor,
    next_edges: Option<Vec<Edge>>,
    input_metadata_: SmallVec<[InputMetaData; 2]>,
}

impl AccumulateGrad {
    pub fn new(tensor: Tensor) -> Self {
        let input_metadata: SmallVec<[_; 2]> = smallvec![InputMetaData::from_tensor(&tensor)];

        Self {
            tensor,
            next_edges: None,
            input_metadata_: input_metadata,
        }
    }

    pub fn call(&mut self, grads: Vec<Tensor>) -> Vec<Tensor> {
        let new_grad = &grads[0];
        let grad = self.tensor.grad();
        if let Some(g) = grad {
            let t = unsafe { &*Rc::into_raw(g) };
            self.tensor.set_grad(t + new_grad);
        } else {
            let t = Tensor::new(new_grad);
            self.tensor.set_grad(t);
        }
        Vec::new()
    }

    // fn set_next_edges(&mut self, _edges: Vec<Edge>) {}
    // fn add_input_metadata(&mut self, _tensor: &Tensor) -> usize {
    //     todo!()
    // }
    // fn next_edges(&self) -> Option<&EdgeList> {
    // None
    // }
    // fn next_edge(&self, _i: usize) -> Option<Edge> {
    // pub input_metadata_: SmallVec<[InputMetaData; 2]>,
    // }
    pub fn num_inputs(&self) -> usize {
        self.input_metadata_.len()
    }
    // fn num_outputs(&self) -> usize {
    //     todo!()
    // }
}

#[derive(Debug)]
pub enum Function {
    AddBackwardTensors(AddBackwardTensors),
    MulBackwardTensors(MulBackwardTensors),
    Add(Add),
}

impl Function {
    pub fn call(&mut self, input: VariableList) -> VariableList {
        match self {
            Function::AddBackwardTensors(f) => f.call(input),
            Function::Add(_) => todo!(),
            Function::MulBackwardTensors(f) => f.call(input),
        }
    }

    pub fn add_input_metadata(&mut self, tensor: &Tensor) -> usize {
        match self {
            Function::AddBackwardTensors(f) => f.add_input_metadata(tensor),
            Function::Add(_) => todo!(),
            Function::MulBackwardTensors(f) => f.add_input_metadata(tensor),
        }
    }

    pub fn set_next_edges(&mut self, edges: Vec<Edge>) {
        match self {
            Function::AddBackwardTensors(f) => f.set_next_edges(edges),
            Function::Add(_) => {}
            Function::MulBackwardTensors(f) => f.set_next_edges(edges),
        }
    }

    pub fn next_edges(&self) -> Option<&EdgeList> {
        match self {
            Function::AddBackwardTensors(f) => f.next_edges(),
            Function::Add(_) => todo!(),
            Function::MulBackwardTensors(f) => f.next_edges(),
        }
    }

    pub fn next_edge(&self, i: usize) -> Option<Edge> {
        match self {
            Function::AddBackwardTensors(f) => f.next_edge(i),
            Function::Add(_) => todo!(),
            Function::MulBackwardTensors(f) => f.next_edge(i),
        }
    }

    pub fn num_inputs(&self) -> usize {
        match self {
            Function::AddBackwardTensors(t) => t.num_inputs(),
            Function::Add(_) => todo!(),
            Function::MulBackwardTensors(f) => f.num_inputs(),
        }
    }

    pub fn num_outputs(&self) -> usize {
        match self {
            Function::AddBackwardTensors(f) => f.num_outputs(),
            Function::Add(_) => todo!(),
            Function::MulBackwardTensors(f) => f.num_outputs(),
        }
    }
}

#[derive(Debug)]
pub struct Add;

#[derive(Debug)]
pub struct InputMetaData {
    pub size: SmallVec<[usize; 4]>,
    pub device: usize,
}

impl InputMetaData {
    pub fn from_tensor(_t: &Tensor) -> InputMetaData {
        InputMetaData {
            size: SmallVec::<[usize; 4]>::new(),
            device: 0,
        }
    }
}

#[derive(Debug)]
pub struct AddBackwardTensors {
    pub input_metadata_: SmallVec<[InputMetaData; 2]>,
    pub next_edges: Option<EdgeList>,
}

impl AddBackwardTensors {
    pub fn call(&mut self, mut grads: Vec<Tensor>) -> Vec<Tensor> {
        let _tmp: Vec<_> = grads.drain(1..).collect();
        let grad = grads.get(0).unwrap().clone();
        let grad_1 = grad.clone();
        let grad_2 = grad.clone();
        vec![grad_1, grad_2]
    }

    pub fn set_next_edges(&mut self, edges: Vec<Edge>) {
        self.next_edges = Some(edges)
    }

    pub fn add_input_metadata(&mut self, tensor: &Tensor) -> usize {
        let input_nr = self.input_metadata_.len();
        self.input_metadata_
            .push(InputMetaData::from_tensor(tensor));
        input_nr
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
        self.input_metadata_.len()
    }

    fn num_outputs(&self) -> usize {
        self.next_edges.as_ref().unwrap().len()
    }
}
#[derive(Debug)]
pub struct MulBackwardTensors {
    pub input_metadata_: SmallVec<[InputMetaData; 2]>,
    pub next_edges: Option<EdgeList>,
    pub _self: Option<SavedTensor>,
    pub other: Option<SavedTensor>,
}

impl MulBackwardTensors {
    pub fn call(&mut self, grads: Vec<Tensor>) -> Vec<Tensor> {
        let grad = grads.first().unwrap();
        let self_ = self._self.as_ref().unwrap().unpack();
        let other = self.other.as_ref().unwrap().unpack();

        let first = &other * grad;
        let second = &self_ * grad;

        vec![first, second]
    }

    pub fn set_next_edges(&mut self, edges: Vec<Edge>) {
        self.next_edges = Some(edges)
    }

    pub fn add_input_metadata(&mut self, tensor: &Tensor) -> usize {
        let input_nr = self.input_metadata_.len();
        self.input_metadata_
            .push(InputMetaData::from_tensor(tensor));
        input_nr
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
        self.input_metadata_.len()
    }

    fn num_outputs(&self) -> usize {
        self.next_edges.as_ref().unwrap().len()
    }
}
