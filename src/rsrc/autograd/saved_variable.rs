use super::{make_variable, make_variable_with_edge};
use crate::ops::*;
use crate::tensor::*;
use crate::util_autograd;
use std::cell::RefCell;
use std::rc::{Rc, Weak};

pub struct SavedTensor {
    data: Tensor,
    grad_accumulator: Option<Weak<RefCell<Node>>>,
    grad_fn: Option<Rc<RefCell<Node>>>,
    saved_version: u32,
    output_nr: usize,
    was_default_constructed: bool,
    requires_grad: bool,
    has_grad_fn: bool,
    version_counter: TensorVersion,
}

impl Default for SavedTensor {
    fn default() -> Self {
        Self {
            data: Tensor::default(),
            was_default_constructed: true,
            grad_accumulator: None,
            grad_fn: None,
            saved_version: 0,
            output_nr: 0,
            requires_grad: false,
            has_grad_fn: false,
            version_counter: TensorVersion::default(),
        }
    }
}

impl SavedTensor {
    pub fn new(tensor: &Tensor, is_output: bool) -> Self {
        if tensor.defined() {
            let was_default_constructed = false;
            let output_nr = tensor.output_nr();
            let requires_grad = tensor.requires_grad();
            let has_grad_fn = !tensor.is_leaf();
            let data = tensor.tensor_data();
            let mut grad_accumulator: Option<Weak<RefCell<Node>>> = None;
            let mut grad_fn = None;
            if tensor.is_leaf() {
                grad_accumulator = util_autograd::grad_accumulator(tensor)
                    .and_then(|acc| Some(Rc::downgrade(&acc)));
            // Some(Rc::downgrade(&util_autograd::grad_accumulator(tensor)));
            } else if !is_output {
                grad_fn = tensor.grad_fn();
            }
            let version_counter = util_autograd::TensorHook::version_counter(tensor).clone();
            let saved_version = version_counter.current_version();
            Self {
                data,
                grad_accumulator,
                grad_fn,
                saved_version,
                output_nr,
                was_default_constructed,
                requires_grad,
                has_grad_fn,
                version_counter,
            }
        } else {
            Self::default()
        }
    }

    pub fn new_with_optional(tensor: Option<&Tensor>, is_output: bool) -> Self {
        let t = match tensor {
            Some(t) => t.clone(),
            None => Tensor::default(),
        };
        Self::new(&t, is_output)
    }

    pub fn unpack(&self) -> Tensor {
        if !self.data.defined() {
            if !self.was_default_constructed {
                panic!("Not sure but pytorch throws error with ERR_BACKWARD_TWICE")
            }
            return Tensor::default();
        }

        if self.saved_version != self.version_counter.current_version() {
            panic!()
        }

        let mut tensor: Tensor;
        if let Some(grad_fn) = &self.grad_fn {
            tensor = make_variable_with_edge(
                self.data.clone(),
                Edge::new(Some(grad_fn.clone()), self.output_nr),
            )
        } else {
            tensor = make_variable(self.data.clone(), self.requires_grad);
        }

        util_autograd::TensorHook::set_version_counter(
            &tensor,
            TensorVersion::new_with_version(self.saved_version),
        );

        if self.requires_grad
            && tensor.grad_fn().is_none()
            && self.grad_accumulator.is_some()
            && self.grad_accumulator.as_ref().unwrap().strong_count() == 0
        {
            panic!()
        }

        util_autograd::TensorHook::set_grad_accumulator(&mut tensor, self.grad_accumulator.clone());
        tensor
    }
}

impl std::fmt::Debug for SavedTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SavedTensor")
            .field("data", &self.data)
            .finish()
    }
}
