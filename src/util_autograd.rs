use crate::autograd::*;
use crate::core::GradMode;
use crate::ops::*;
use crate::tensor::*;
use std::cell::RefCell;
use std::rc::Rc;
use std::rc::Weak;

pub fn compute_requires_grad(tensors: &[&Tensor]) -> bool {
    let mut out = false;
    if !GradMode::is_enabled() {
        out
    } else {
        for t in tensors {
            if t.requires_grad() {
                out = true;
            }
        }
        out
    }
}

pub struct TensorHook;
impl TensorHook {
    pub fn grad_fn(tensor: &Tensor) -> Option<&Rc<RefCell<Node>>> {
        if let Some(meta) = Self::get_autograd_meta(tensor) {
            meta.grad_fn_.as_ref()
        } else {
            None
        }
    }

    pub fn get_autograd_meta(tensor: &Tensor) -> Option<&mut AutogradMeta> {
        let w = unsafe { &mut *tensor._impl.clone().as_ptr() };
        w.get_autogradmeta()
    }

    // Todo: Use get_autograd_meta here instead of set_gradient_edge()
    pub fn materialize_autograd_meta(tensor: &Tensor) -> &mut AutogradMeta {
        let mut p = tensor._impl.borrow_mut();
        if p.autogradmeta.as_ref().is_none() {
            p.set_autograd_meta(Some(AutogradMetaFactory::make()))
        }
        TensorHook::get_autograd_meta(tensor).unwrap()
    }

    pub fn tensor_data(tensor: &Tensor) -> Tensor {
        let tensor_impl_copy = tensor
            .get_unsafe_tensor_impl()
            .shallow_copy_and_detach(tensor.get_unsafe_tensor_impl().version_counter());
        Tensor::from_impl(tensor_impl_copy)
    }

    pub fn version_counter(tensor: &Tensor) -> &TensorVersion {
        tensor.get_unsafe_tensor_impl().version_counter()
    }

    pub fn set_grad_accumulator(tensor: &Tensor, grad_accumulator: Option<Weak<RefCell<Node>>>) {
        let t = Self::materialize_autograd_meta(tensor);
        t.grad_accumulator_ = grad_accumulator;
    }

    pub fn set_version_counter(tensor: &Tensor, version_counter: TensorVersion) {
        let impl_ = tensor.get_unsafe_tensor_impl();
        impl_.set_version_counter(version_counter);
    }
}

pub fn grad_accumulator(tensor: &Tensor) -> Option<Rc<RefCell<Node>>> {
    if let Some(meta) = TensorHook::get_autograd_meta(tensor) {
        if let Some(acc_) = meta.grad_accumulator_.as_ref() {
            Weak::upgrade(acc_)
        } else {
            let result = Rc::new(RefCell::new(Node::new(AccumulateGrad::new(
                Tensor::new(tensor),
            ))));
            meta.grad_accumulator_ = Some(Rc::downgrade(&result));
            Some(result)
        }
    } else {
        None
    }
}
pub fn gradient_edge(tensor: &Tensor) -> Edge {
    if let Some(grad_fn) = tensor.grad_fn() {
        Edge::new(Some(grad_fn.clone()), tensor.output_nr())
    } else {
        Edge::new(grad_accumulator(tensor), 0)
    }
}

pub fn collect_next_edges(tensors: &[&Tensor]) -> Vec<Edge> {
    let mut next_edges: Vec<Edge> = vec![];
    next_edges.reserve(tensors.len());
    for t in tensors {
        next_edges.push(gradient_edge(*t));
    }
    next_edges
}

pub fn set_gradient_edge(tensor: &Tensor, args: (Rc<RefCell<Node>>, usize)) {
    let edge = Edge::new(Some(args.0), args.1);
    // Todo: read todo on materialize_autograd_meta
    TensorHook::materialize_autograd_meta(tensor);

    let mut meta = tensor._impl.borrow_mut();
    meta.set_grad_fn(edge.function);
    meta.set_output_nr(edge.input_nr);
}

pub fn set_history(tensor: &Tensor, grad_fn: Rc<RefCell<Node>>) {
    let output_nr = grad_fn.borrow_mut().add_input_metadata(tensor);
    set_gradient_edge(tensor, (grad_fn, output_nr))
}
