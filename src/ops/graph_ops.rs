use crate::{
    aten::native::{self, log_softmax_backward_cpu, sigmoid_backward},
    c10::{Scalar, ScalarType},
    ops::NodeTrait,
    tensor::*,
    util::index_generator::IndexGenerator,
};
use crate::{autograd::*, util::BitSet};
use loss::Reduction;
use smallvec::*;

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
        let new_grad = &grads[0];
        let grad = self.tensor.grad();
        // if self.tensor.ndimension() == 1 {
        //     println!(
        //         "calling AccumulateGrad for Tensor: {:?},\n{:?}",
        //         self.tensor, new_grad
        //     );
        // }
        if let Some(g) = grad {
            self.tensor.set_grad(&g + new_grad);
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

    fn input_metadata(&self, index: usize) -> &InputMetaData {
        self.input_metadata_.get(index).unwrap()
    }

    fn debug_print(&self) -> String {
        "Accumulate_Grad".to_string()
    }
}

#[derive(Debug)]
pub struct Add;

#[derive(Debug)]
pub struct InputMetaData {
    pub size: SmallVec<[usize; 5]>,
    pub device: usize,
}

impl InputMetaData {
    fn from_tensor(t: &Tensor) -> InputMetaData {
        InputMetaData {
            size: SmallVec::from_slice(t.sizes()),
            device: 0,
        }
    }

    pub fn shape(&self) -> &[usize] {
        self.size.as_slice()
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

    fn input_metadata(&self, index: usize) -> &InputMetaData {
        self.input_metadata_.get(index).unwrap()
    }
    fn debug_print(&self) -> String {
        "AddBackwardTensors".to_string()
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

    fn input_metadata(&self, index: usize) -> &InputMetaData {
        self.input_metadata_.get(index).unwrap()
    }

    fn debug_print(&self) -> String {
        "MulBackwardTensors".to_string()
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

    fn input_metadata(&self, index: usize) -> &InputMetaData {
        self.input_metadata_.get(index).unwrap()
    }

    fn debug_print(&self) -> String {
        "AddBackwardScalar".to_string()
    }
}

pub struct MulBackwardScalar {
    pub input_metadata_: SmallVec<[InputMetaData; 1]>,
    pub next_edges: Option<EdgeList>,
    pub _self: Option<SavedTensor>,
    pub other: Scalar,
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

    fn input_metadata(&self, index: usize) -> &InputMetaData {
        self.input_metadata_.get(index).unwrap()
    }

    fn debug_print(&self) -> String {
        "MulBackwardScalar".to_string()
    }
}

pub struct SubBackwardTensors {
    pub input_metadata_: SmallVec<[InputMetaData; 2]>,
    pub next_edges: Option<EdgeList>,
}

impl NodeTrait for SubBackwardTensors {
    fn call(&mut self, mut grads: Vec<Tensor>) -> Vec<Tensor> {
        let _tmp: Vec<_> = grads.drain(1..).collect();
        let mut gen = IndexGenerator::new();
        let self_idx = gen.next();
        let other_idx = gen.next();
        let mut grad_inputs = Vec::with_capacity(gen.len());
        let grad = grads.get(0).unwrap().clone();
        if self.should_compute_output(self_idx) {
            grad_inputs.push(grad.clone())
        }
        if self.should_compute_output(other_idx) {
            grad_inputs.push(-&grad.clone())
        }
        grad_inputs
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

    fn input_metadata(&self, index: usize) -> &InputMetaData {
        self.input_metadata_.get(index).unwrap()
    }

    fn debug_print(&self) -> String {
        "SubBackwardTensors".to_string()
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

    fn input_metadata(&self, index: usize) -> &InputMetaData {
        self.input_metadata_.get(index).unwrap()
    }

    fn debug_print(&self) -> String {
        "DivBackwardTensors".to_string()
    }
}

pub struct DivBackwardScalar {
    pub input_metadata_: SmallVec<[InputMetaData; 1]>,
    pub next_edges: Option<EdgeList>,
    pub _self: Option<SavedTensor>,
    pub other: Scalar,
}

impl NodeTrait for DivBackwardScalar {
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

    fn input_metadata(&self, index: usize) -> &InputMetaData {
        self.input_metadata_.get(index).unwrap()
    }

    fn debug_print(&self) -> String {
        "DivBackwardScalar".to_string()
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

    fn input_metadata(&self, index: usize) -> &InputMetaData {
        self.input_metadata_.get(index).unwrap()
    }

    fn debug_print(&self) -> String {
        "NegBackward".to_string()
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

    fn input_metadata(&self, index: usize) -> &InputMetaData {
        self.input_metadata_.get(index).unwrap()
    }

    fn debug_print(&self) -> String {
        "TBackward".to_string()
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

fn maybe_mutliply(t: Tensor, alpha: Scalar) -> Tensor {
    let mut is_one = false;
    if alpha.is_floating_point() {
        let one: f64 = alpha.to();
        is_one = one == 1.0;
    } else if alpha.is_integer() {
        let one: i64 = alpha.to();
        is_one = one == 1;
    }
    if is_one {
        t
    } else {
        t * alpha
    }
}

fn mm_mat1_backward(
    grad: &Tensor,
    mat2: &Tensor,
    mat1: &Tensor,
    alpha: impl Into<Scalar>,
) -> Tensor {
    let sizes = mat1.sizes();
    let strides = mat1.strides();
    if strides[0] == 1 && strides[1] == sizes[0] {
        return maybe_mutliply(mat2.mm(grad.t(), true), alpha.into());
    }
    maybe_mutliply(grad.mm(mat2.t(), true), alpha.into())
}

fn mm_mat2_backward(
    grad: &Tensor,
    mat1: &Tensor,
    sizes: &[usize],
    strides: &[usize],
    alpha: impl Into<Scalar>,
) -> Tensor {
    if strides[0] == 1 && strides[1] == sizes[0] {
        return maybe_mutliply(grad.t().mm(mat1, false).t(), alpha.into());
    }
    maybe_mutliply(mat1.t().mm(grad, false), alpha.into())
}

impl NodeTrait for MmBackward {
    fn call(&mut self, grads: Vec<Tensor>) -> Vec<Tensor> {
        let grad = grads.first().unwrap();
        let mat1 = self.self_.as_ref().unwrap().unpack();
        let mat2 = self.mat2_.as_ref().unwrap().unpack();
        //Todo: imlement and use mm_mat2_backward and mm_mat1_backward
        let mat2_grad =
            mm_mat2_backward(grad, &mat1, self.mat2_sizes.as_slice(), mat2.strides(), 1);
        // mat1.t().mm(grad, false);
        let mat1_grad = mm_mat1_backward(grad, &mat2, &mat1, 1);
        // grad.mm(&mat2.t(), false);
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

    fn input_metadata(&self, index: usize) -> &InputMetaData {
        self.input_metadata_.get(index).unwrap()
    }

    fn debug_print(&self) -> String {
        "MmBackward".to_string()
    }
}

pub struct AddmmBackward {
    pub input_metadata_: SmallVec<[InputMetaData; 2]>,
    pub next_edges: Option<EdgeList>,

    pub mat1_: Option<SavedTensor>,
    pub mat2_: Option<SavedTensor>,
    pub mat2_sizes: Vec<usize>,
    pub alpha: Scalar,
    pub beta: Scalar,
}

impl NodeTrait for AddmmBackward {
    fn call(&mut self, grads: Vec<Tensor>) -> Vec<Tensor> {
        let grad = grads.first().unwrap();
        let mat1 = self.mat1_.as_ref().unwrap().unpack();
        let mat2 = self.mat2_.as_ref().unwrap().unpack();
        let mut gen = IndexGenerator::new();
        let self_idx = gen.next();
        let mat1_idx = gen.next();
        let mat2_idx = gen.next();
        let mut grad_inputs = Vec::with_capacity(gen.len());
        if self.should_compute_output(self_idx) {
            let self_grad = maybe_mutliply(grad.clone(), self.beta);
            grad_inputs.push(self_grad);
        }
        if self.should_compute_output(mat1_idx) {
            let mat1_grad = mm_mat1_backward(grad, &mat2, &mat1, self.alpha);
            grad_inputs.push(mat1_grad);
        }
        if self.should_compute_output(mat2_idx) {
            let mat2_grad = mm_mat2_backward(
                grad,
                &mat1,
                self.mat2_sizes.as_slice(),
                mat2.strides(),
                self.alpha,
            );
            grad_inputs.push(mat2_grad);
        }
        grad_inputs
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

    fn input_metadata(&self, index: usize) -> &InputMetaData {
        self.input_metadata_.get(index).unwrap()
    }

    fn debug_print(&self) -> String {
        "AddMmBackward".to_string()
    }
}

pub struct SigmoidBackward {
    pub input_metadata_: SmallVec<[InputMetaData; 2]>,
    pub next_edges: Option<EdgeList>,
    pub result_: Option<SavedTensor>,
}

impl NodeTrait for SigmoidBackward {
    fn call(&mut self, input: Vec<Tensor>) -> Vec<Tensor> {
        let grad_input = input.first().unwrap();
        let result = self.result_.as_ref().unwrap().unpack();
        let grad_result = sigmoid_backward(grad_input, &result);
        // dbg!(&grad_result);
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

    fn input_metadata(&self, index: usize) -> &InputMetaData {
        self.input_metadata_.get(index).unwrap()
    }

    fn debug_print(&self) -> String {
        "SigmoidBackward".to_string()
    }
}
pub struct SumBackward0 {
    pub input_metadata_: SmallVec<[InputMetaData; 2]>,
    pub next_edges: Option<EdgeList>,
    pub self_sizes: Vec<usize>,
}

impl NodeTrait for SumBackward0 {
    fn call(&mut self, grads: Vec<Tensor>) -> Vec<Tensor> {
        let grad = grads.first().unwrap();
        let grad_result = grad.expand(self.self_sizes.as_slice(), false);
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

    fn input_metadata(&self, index: usize) -> &InputMetaData {
        self.input_metadata_.get(index).unwrap()
    }

    fn debug_print(&self) -> String {
        "SigmoidBackward".to_string()
    }
}
const dim_bitset_size: usize = 64;

#[inline(always)]
fn dim_list_to_bitset(dims: &[usize], ndims: i64) -> BitSet<dim_bitset_size> {
    assert!(
        ndims <= dim_bitset_size as i64,
        "only tensors with up to {} dims are supported",
        dim_bitset_size
    );
    let mut seen = BitSet::<dim_bitset_size>::default();
    for i in 0..dims.len() {
        let dim = maybe_wrap_dim(dims[i] as i64, ndims, false);
        assert!(
            !seen.check(dim),
            "dim {} appears multiple times in the list of dims",
            dim
        );
        seen.assign(dim, true);
    }
    return seen;
}

fn unsqueeze_multiple(t: &Tensor, dim: &[usize], n_dims: usize) -> Tensor {
    let dims_to_unsqueeze = dim_list_to_bitset(dim, n_dims as i64);
    let mut res = t.clone();
    for i in 0..n_dims {
        if dims_to_unsqueeze.check(i) {
            res = res.unsqueeze(i);
        }
    }
    return res;
}

fn sum_backward(grad: &Tensor, sizes: &[usize], dims: &[usize], keepdim: bool) -> Tensor {
    if !keepdim && sizes.len() > 0 {
        if dims.len() == 1 {
            return grad.unsqueeze(dims[0]).expand(sizes, false);
        } else {
            let res = unsqueeze_multiple(grad, dims, sizes.len());
            return res.expand(sizes, false);
        }
    } else {
        return grad.expand(sizes, false);
    }
}
pub struct SumBackward1 {
    pub input_metadata_: SmallVec<[InputMetaData; 2]>,
    pub next_edges: Option<EdgeList>,
    pub dim: Vec<usize>,
    pub self_sizes: Vec<usize>,
    pub keep_dim: bool,
}

impl NodeTrait for SumBackward1 {
    fn call(&mut self, grads: Vec<Tensor>) -> Vec<Tensor> {
        let grad = grads.first().unwrap();
        let grad_result = sum_backward(
            grad,
            self.self_sizes.as_slice(),
            self.dim.as_slice(),
            self.keep_dim,
        );
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

    fn input_metadata(&self, index: usize) -> &InputMetaData {
        self.input_metadata_.get(index).unwrap()
    }

    fn debug_print(&self) -> String {
        "SigmoidBackward".to_string()
    }
}

pub struct BinaryCrossEntropyBackward {
    pub input_metadata_: SmallVec<[InputMetaData; 2]>,
    pub next_edges: Option<EdgeList>,
    pub self_: Option<SavedTensor>,
    pub target_: Option<SavedTensor>,
    pub weight_: Option<SavedTensor>,
    pub reduction: loss::Reduction,
}

impl Default for BinaryCrossEntropyBackward {
    fn default() -> Self {
        Self {
            input_metadata_: smallvec![],
            next_edges: None,
            self_: None,
            target_: None,
            weight_: None,
            reduction: loss::Reduction::None,
        }
    }
}

impl NodeTrait for BinaryCrossEntropyBackward {
    fn call(&mut self, input: Vec<Tensor>) -> Vec<Tensor> {
        let grad_input = input.first().unwrap();
        let self_ = self.self_.as_ref().unwrap().unpack();
        let target_ = self.target_.as_ref().unwrap().unpack();
        let weight = self.weight_.as_ref().unwrap().unpack();
        let grad_result = loss::binary_cross_entropy_backward(
            grad_input,
            &self_,
            &target_,
            &weight,
            self.reduction,
        );
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

    fn input_metadata(&self, index: usize) -> &InputMetaData {
        self.input_metadata_.get(index).unwrap()
    }

    fn debug_print(&self) -> String {
        "BinaryCrossEntropyBackward".to_string()
    }
}

pub struct MeanBackward {
    pub input_metadata_: SmallVec<[InputMetaData; 2]>,
    pub next_edges: Option<EdgeList>,
    pub self_sizes: Vec<usize>,
    pub self_numel: usize,
    pub self_scalar_type: ScalarType,
}

impl NodeTrait for MeanBackward {
    fn call(&mut self, input: Vec<Tensor>) -> Vec<Tensor> {
        let grad_input = input.first().unwrap();
        let mut result = grad_input.expand(self.self_sizes.as_slice(), false);
        result.div_(&full(result.sizes(), self.self_numel as f32, None));
        vec![result]
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

    fn input_metadata(&self, index: usize) -> &InputMetaData {
        self.input_metadata_.get(index).unwrap()
    }

    fn debug_print(&self) -> String {
        "MeanBackward".to_string()
    }
}

pub struct LogSoftmaxBackward {
    pub input_metadata_: SmallVec<[InputMetaData; 2]>,
    pub next_edges: Option<EdgeList>,
    pub self_: Option<SavedTensor>,
    pub result: Option<SavedTensor>,
    pub dim: i64,
}

impl Default for LogSoftmaxBackward {
    fn default() -> Self {
        Self {
            input_metadata_: smallvec![],
            next_edges: None,
            self_: None,
            result: None,
            dim: 0,
        }
    }
}

impl NodeTrait for LogSoftmaxBackward {
    fn call(&mut self, input: Vec<Tensor>) -> Vec<Tensor> {
        let grad_input = input.first().unwrap();
        let result = self.result.as_ref().unwrap().unpack();
        let grad_result = log_softmax_backward_cpu(grad_input, &result, self.dim);
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

    fn input_metadata(&self, index: usize) -> &InputMetaData {
        self.input_metadata_.get(index).unwrap()
    }

    fn debug_print(&self) -> String {
        "LogSoftmaxBackward".to_string()
    }
}
pub struct NllLossBackward {
    pub input_metadata_: SmallVec<[InputMetaData; 2]>,
    pub next_edges: Option<EdgeList>,
    pub self_: Option<SavedTensor>,
    pub weight: Option<SavedTensor>,
    pub target: Option<SavedTensor>,
    pub total_weight: Option<SavedTensor>,
    pub reduction: Reduction,
    pub ignore_index: i64,
}
impl Default for NllLossBackward {
    fn default() -> Self {
        Self {
            input_metadata_: smallvec![],
            next_edges: None,
            ignore_index: 0,
            reduction: Reduction::Mean,
            self_: None,
            target: None,
            weight: None,
            total_weight: None,
        }
    }
}

impl NodeTrait for NllLossBackward {
    fn call(&mut self, input: Vec<Tensor>) -> Vec<Tensor> {
        let grad_input = input.first().unwrap();
        let self_ = self.self_.as_ref().unwrap().unpack();
        let weight = self.weight.as_ref().unwrap().unpack();
        let target = self.target.as_ref().unwrap().unpack();
        let total_weight = self.total_weight.as_ref().unwrap().unpack();
        let grad_result = native::nll_loss_backward_cpu(
            grad_input,
            &self_,
            &target,
            &weight,
            self.reduction,
            self.ignore_index,
            &total_weight,
        );
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

    fn input_metadata(&self, index: usize) -> &InputMetaData {
        self.input_metadata_.get(index).unwrap()
    }

    fn debug_print(&self) -> String {
        "NllLossBackward".to_string()
    }
}
