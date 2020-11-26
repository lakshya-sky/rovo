use crate::ops::*;
use crate::tensor::*;
use std::cell::RefCell;
use std::rc::{Rc, Weak};

pub struct AutogradMeta {
    pub grad_fn_: Option<Rc<RefCell<Node>>>,
    pub grad_accumulator_: Option<Weak<RefCell<Node>>>,
    pub grad_: Option<Tensor>,
    pub requires_grad: bool,
    pub output_nr: usize,
}

impl AutogradMeta {
    pub fn new(_impl_: &TensorImpl, requires_grad: bool, edge: Edge) -> AutogradMeta {
        let grad_fn = edge.function;
        let output_nr = edge.input_nr;
        // Todo: pytorch has extra function to set requires_grad.
        // See: https://github.com/pytorch/pytorch/blob/115d226498cd37358af187ac54c8c76ddd5df5ed/torch/csrc/autograd/variable.h#L251
        AutogradMeta {
            grad_fn_: grad_fn,
            grad_accumulator_: None,
            grad_: None,
            requires_grad,
            output_nr,
        }
    }
    pub fn new_without_edge(impl_: &TensorImpl, requires_grad: bool) -> AutogradMeta {
        let edge = Edge::empty();
        Self::new(impl_, requires_grad, edge)
    }
    pub fn grad(&self) -> Option<Tensor> {
        self.grad_.clone()
    }
    pub fn set_grad(&mut self, grad: Tensor) {
        self.grad_ = Some(grad)
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad || self.grad_fn_.is_some()
    }

    pub fn grad_fn(&self) -> Option<Rc<RefCell<Node>>> {
        if let Some(fn_) = self.grad_fn_.as_ref() {
            let clone = (*fn_).clone();
            Some(clone)
        // todo!()
        } else {
            None
        }
    }

    pub fn set_grad_fn(&mut self, grad_fn: Option<Rc<RefCell<Node>>>) {
        self.grad_fn_ = grad_fn;
    }
    pub fn set_output_nr(&mut self, output_nr: usize) {
        self.output_nr = output_nr;
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad
    }

    pub fn grad_fn_reset(&mut self) {
        if let Some(grad_fn) = self.grad_fn_.as_ref() {
            drop(grad_fn);
            self.grad_fn_ = None;
        }
    }
}

pub struct AutogradMetaFactory {}

impl AutogradMetaFactory {
    pub fn make() -> AutogradMeta {
        let edge = Edge::empty();
        AutogradMeta {
            grad_: None,
            grad_fn_: edge.function,
            grad_accumulator_: None,
            requires_grad: false,
            output_nr: edge.input_nr,
        }
    }
}
