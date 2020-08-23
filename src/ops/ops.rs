use crate::autograd::SavedTensor;
use crate::ops::NodeTrait;
use crate::tensor::*;
use smallvec::*;
use std::rc::Rc;

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
}

impl NodeTrait for AccumulateGrad {
    fn call(&mut self, grads: Vec<Tensor>) -> Vec<Tensor> {
        // println!(
        //     "calling AccumulateGrad for Tensor: {:?}",
        //     self.tensor
        // );
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

    fn set_next_edges(&mut self, _edges: Vec<Edge>) {}
    fn add_input_metadata(&mut self, _tensor: &Tensor) -> usize {
        todo!()
    }
    fn next_edges(&self) -> Option<&EdgeList> {
        None
    }
    fn next_edge(&self, _i: usize) -> Option<Edge> {
        None
    }
    fn num_inputs(&self) -> usize {
        self.input_metadata_.len()
    }
    fn num_outputs(&self) -> usize {
        todo!()
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
    fn from_tensor(_t: &Tensor) -> InputMetaData {
        InputMetaData {
            size: SmallVec::<[usize; 4]>::new(),
            device: 0,
        }
    }
}

pub struct AddBackwardTensors {
    pub input_metadata_: SmallVec<[InputMetaData; 2]>,
    pub next_edges: Option<EdgeList>,
}

impl NodeTrait for AddBackwardTensors {
    fn call(&mut self, mut grads: Vec<Tensor>) -> Vec<Tensor> {
        let _tmp: Vec<_> = grads.drain(1..).collect();
        let grad = grads.get(0).unwrap().clone();
        let grad_1 = grad.clone();
        let grad_2 = grad.clone();
        vec![grad_1, grad_2]
    }

    fn set_next_edges(&mut self, edges: Vec<Edge>) {
        self.next_edges = Some(edges)
    }

    fn add_input_metadata(&mut self, tensor: &Tensor) -> usize {
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
pub struct MulBackwardTensors {
    pub input_metadata_: SmallVec<[InputMetaData; 2]>,
    pub next_edges: Option<EdgeList>,
    pub _self: Option<SavedTensor>,
    pub other: Option<SavedTensor>,
}

impl NodeTrait for MulBackwardTensors {
    fn call(&mut self, grads: Vec<Tensor>) -> Vec<Tensor> {
        let grad = grads.first().unwrap();
        let self_ = self._self.as_ref().unwrap().unpack();
        let other = self.other.as_ref().unwrap().unpack();

        let first = &other * grad;
        let second = &self_ * grad;

        vec![first, second]
    }

    fn set_next_edges(&mut self, edges: Vec<Edge>) {
        self.next_edges = Some(edges)
    }

    fn add_input_metadata(&mut self, tensor: &Tensor) -> usize {
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
pub struct AddBackwardScalar {
    pub input_metadata_: SmallVec<[InputMetaData; 1]>,
    pub next_edges: Option<EdgeList>,
}

impl NodeTrait for AddBackwardScalar {
    fn call(&mut self, grads: Vec<Tensor>) -> Vec<Tensor> {
        let grad = grads.first().unwrap();
        vec![grad.clone()]
    }

    fn set_next_edges(&mut self, edges: Vec<Edge>) {
        self.next_edges = Some(edges)
    }

    fn add_input_metadata(&mut self, tensor: &Tensor) -> usize {
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

pub struct MulBackwardScalar {
    pub input_metadata_: SmallVec<[InputMetaData; 1]>,
    pub next_edges: Option<EdgeList>,
    pub _self: Option<SavedTensor>,
    pub other: f64,
}

impl NodeTrait for MulBackwardScalar {
    fn call(&mut self, grads: Vec<Tensor>) -> Vec<Tensor> {
        let grad = grads.first().unwrap();
        let other = self.other;

        let first = grad * other;

        vec![first]
    }

    fn set_next_edges(&mut self, edges: Vec<Edge>) {
        self.next_edges = Some(edges)
    }

    fn add_input_metadata(&mut self, tensor: &Tensor) -> usize {
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

pub struct SubBackwardTensors {
    pub input_metadata_: SmallVec<[InputMetaData; 2]>,
    pub next_edges: Option<EdgeList>,
}

impl NodeTrait for SubBackwardTensors {
    fn call(&mut self, mut grads: Vec<Tensor>) -> Vec<Tensor> {
        let _tmp: Vec<_> = grads.drain(1..).collect();
        let grad = grads.get(0).unwrap().clone();
        let grad_1 = grad.clone();
        let grad_2 = -&grad.clone();
        vec![grad_1, grad_2]
    }

    fn set_next_edges(&mut self, edges: Vec<Edge>) {
        self.next_edges = Some(edges)
    }

    fn add_input_metadata(&mut self, tensor: &Tensor) -> usize {
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

pub struct DivBackwardTensors {
    pub input_metadata_: SmallVec<[InputMetaData; 2]>,
    pub next_edges: Option<EdgeList>,
    pub _self: Option<SavedTensor>,
    pub other: Option<SavedTensor>,
}

impl NodeTrait for DivBackwardTensors {
    fn call(&mut self, grads: Vec<Tensor>) -> Vec<Tensor> {
        let grad = grads.first().unwrap();
        let self_ = self._self.as_ref().unwrap().unpack();
        let other = self.other.as_ref().unwrap().unpack();

        let first = grad / &other;
        let second = -grad * &self_ / (&other * &other);

        vec![first, second]
    }

    fn set_next_edges(&mut self, edges: Vec<Edge>) {
        self.next_edges = Some(edges)
    }

    fn add_input_metadata(&mut self, tensor: &Tensor) -> usize {
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

pub struct NegBackward {
    pub input_metadata_: SmallVec<[InputMetaData; 2]>,
    pub next_edges: Option<EdgeList>,
}

impl NodeTrait for NegBackward {
    fn call(&mut self, input: Vec<Tensor>) -> Vec<Tensor> {
        let grad_input = input.first().unwrap();
        let grad_result = -grad_input;
        vec![grad_result]
    }

    fn set_next_edges(&mut self, edges: Vec<Edge>) {
        self.next_edges = Some(edges)
    }

    fn add_input_metadata(&mut self, tensor: &Tensor) -> usize {
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

pub struct TBackward {
    pub input_metadata_: SmallVec<[InputMetaData; 2]>,
    pub next_edges: Option<EdgeList>,
}

impl NodeTrait for TBackward {
    fn call(&mut self, input: Vec<Tensor>) -> Vec<Tensor> {
        let grad_input = input.first().unwrap();
        let grad_result = grad_input.t();
        vec![grad_result]
    }

    fn set_next_edges(&mut self, edges: Vec<Edge>) {
        self.next_edges = Some(edges)
    }

    fn add_input_metadata(&mut self, tensor: &Tensor) -> usize {
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

impl std::fmt::Debug for TBackward {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "TBackward")
    }
}

pub struct MmBackward {
    pub input_metadata_: SmallVec<[InputMetaData; 2]>,
    pub next_edges: Option<EdgeList>,
    pub mat2_sizes: Vec<usize>,
    pub self_: Option<SavedTensor>,
    pub mat2_: Option<SavedTensor>,
}

impl NodeTrait for MmBackward {
    fn call(&mut self, grads: Vec<Tensor>) -> Vec<Tensor> {
        let grad = grads.first().unwrap();
        let mat1 = self.self_.as_ref().unwrap().unpack();
        let mat2 = self.mat2_.as_ref().unwrap().unpack();
        //Todo: imlement and use mm_mat2_backward and mm_mat1_backward

        // println!("Mat1: {:?}", mat1);
        // println!("grad: {:?}", grad);
        let mat2_grad = mat1.t().mm(grad, false);
        let mat1_grad = grad.mm(&mat2.t(), false);
        // println!("Mat 2 Grad: {:?}", mat2_grad);
        vec![mat1_grad, mat2_grad]
    }

    fn set_next_edges(&mut self, edges: Vec<Edge>) {
        self.next_edges = Some(edges)
    }

    fn add_input_metadata(&mut self, tensor: &Tensor) -> usize {
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

impl std::fmt::Debug for MmBackward {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "MmBackward {:?}", self.mat2_)
    }
}
