// use crate::tensor::*;
// use smallvec::SmallVec;
// use std::rc::Rc;

// pub trait Node {
//     fn call(self, input: Vec<Tensor>) -> Vec<Tensor>;
//     fn set_next_edges(&mut self, edges: Vec<Edge>);
//     fn add_input_metadata(&mut self, tensor: &Tensor) -> usize;
//     fn next_edges(&self) -> Option<&EdgeList>;
//     fn next_edge(&self, i: usize) -> Option<&Edge>;
//     fn num_inputs(&self) -> usize;
//     fn num_outputs(&self) -> usize;
// }

// pub trait Function: Node {}

// pub struct AccumulateGrad {
//     tensor: Tensor,
//     next_edges: Option<Vec<Edge>>,
// }

// impl AccumulateGrad {
//     pub fn new(tensor: Tensor) -> Self {
//         Self {
//             tensor,
//             next_edges: None,
//         }
//     }
// }

// impl Node for AccumulateGrad {
//     fn call(mut self, grads: Vec<Tensor>) -> Vec<Tensor> {
//         let new_grad = &grads[0];
//         let grad = self.tensor.grad();
//         if let Some(g) = grad {
//             let t = unsafe { &*Rc::into_raw(g) };
//             self.tensor.set_grad(t + new_grad);
//         } else {
//             let t = Tensor::new(new_grad);
//             self.tensor.set_grad(t);
//         }
//         Vec::new()
//     }
//     fn set_next_edges(&mut self, _edges: Vec<Edge>) {}
//     fn add_input_metadata(&mut self, _tensor: &Tensor) -> usize {
//         todo!()
//     }
//     fn next_edges(&self) -> Option<&EdgeList> {
//         None
//     }
//     fn next_edge(&self, _i: usize) -> Option<&Edge> {
//         todo!()
//     }
//     fn num_inputs(&self) -> usize {
//         todo!()
//     }
//     fn num_outputs(&self) -> usize {
//         todo!()
//     }
// }

// pub struct InputMetaData {
//     pub size: SmallVec<[usize; 4]>,
//     pub device: usize,
// }

// impl InputMetaData {
//     pub fn from_tensor(_t: &Tensor) -> InputMetaData {
//         InputMetaData {
//             size: SmallVec::<[usize; 4]>::new(),
//             device: 0,
//         }
//     }
// }

// pub struct AddBackwardTensors {
//     pub input_metadata_: SmallVec<[InputMetaData; 2]>,
//     pub next_edges: Option<EdgeList>,
// }

// impl Node for AddBackwardTensors {
//     fn call(self, mut grads: Vec<Tensor>) -> Vec<Tensor> {
//         let _tmp: Vec<_> = grads.drain(1..).collect();
//         let grad = grads.pop().unwrap();
//         vec![grad]
//     }
//     fn set_next_edges(&mut self, edges: Vec<Edge>) {
//         self.next_edges = Some(edges)
//     }

//     fn add_input_metadata(&mut self, tensor: &Tensor) -> usize {
//         let input_nr = self.input_metadata_.len();
//         self.input_metadata_
//             .push(InputMetaData::from_tensor(tensor));
//         input_nr
//     }
//     fn next_edges(&self) -> Option<&EdgeList> {
//         self.next_edges.as_ref()
//     }
//     fn next_edge(&self, _i: usize) -> Option<&Edge> {
//         todo!()
//     }
//     fn num_inputs(&self) -> usize {
//         self.input_metadata_.len()
//     }
//     fn num_outputs(&self) -> usize {
//         self.next_edges.as_ref().unwrap().len()
//     }
// }
