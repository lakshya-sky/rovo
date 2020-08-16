use crate::autograd::*;
use crate::ops::*;
use crate::tensor::*;
use crate::util::TensorHook;
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone, Debug)]
pub struct Tensor {
    pub _impl: Rc<RefCell<TensorImpl>>,
}

impl Tensor {
    pub fn from_impl(_impl: TensorImpl) -> Tensor {
        Tensor {
            _impl: Rc::new(RefCell::new(_impl)),
        }
    }

    pub fn new(other: &Tensor) -> Tensor {
        Tensor {
            _impl: other._impl.clone(),
        }
    }

    pub fn make_variable(other: Tensor, gradient_edge: Edge) -> Self {
        let mut other_impl_copy = other._impl.borrow().shallow_copy();
        other_impl_copy.set_autograd_meta(Some(AutogradMeta::new(
            &other_impl_copy,
            false,
            gradient_edge,
        )));
        Self::from_impl(other_impl_copy)
    }

    pub fn make_variable_without_edge(other: Tensor, requires_grad: bool) -> Self {
        let other_tensor_impl = &other._impl;
        if Rc::strong_count(other_tensor_impl) == 1 && other_tensor_impl.borrow().unique_version() {
            todo!()
        } else {
            let mut other_impl_copy = other_tensor_impl.borrow().shallow_copy();
            if requires_grad {
                other_impl_copy.set_autograd_meta(Some(AutogradMeta::new_without_edge(
                    &other_impl_copy,
                    requires_grad,
                )))
            } else {
                other_impl_copy.set_autograd_meta(None);
            }
            Tensor::from_impl(other_impl_copy)
        }
    }

    pub fn from_scalar(shape: &[usize], scalar: f64, requires_grad: bool) -> Tensor {
        Tensor {
            _impl: Rc::new(RefCell::new(TensorImpl::from_scalar(
                shape,
                scalar,
                requires_grad,
            ))),
        }
    }

    pub fn ones(shape: &[usize]) -> Tensor {
        Tensor {
            _impl: Rc::new(RefCell::new(TensorImpl::ones(shape))),
        }
    }

    pub fn ones_like(other: &Tensor) -> Tensor {
        Tensor {
            _impl: Rc::new(RefCell::new(TensorImpl::ones(other.shape()))),
        }
    }

    pub fn shape(&self) -> &[usize] {
        let t = self._impl.clone();
        let q = unsafe { &*t.as_ptr() };
        q.data.shape()
    }

    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor {
            _impl: Rc::new(RefCell::new(TensorImpl::zeros(shape))),
        }
    }

    pub fn grad(&self) -> Option<Rc<Tensor>> {
        self._impl.borrow().grad()
    }

    pub fn set_grad(&mut self, other: Tensor) {
        self._impl.borrow_mut().set_grad(other);
    }

    pub fn requires_grad(&self) -> bool {
        self._impl.borrow().requires_grad()
    }

    pub fn grad_fn(&self) -> Option<Rc<RefCell<Node>>> {
        let t = self._impl.borrow();
        if let Some(p) = t.autogradmeta.as_ref() {
            if let Some(grad_fn) = p.grad_fn() {
                Some(grad_fn.clone())
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn output_nr(&self) -> usize {
        if let Some(meta) = self._impl.borrow().autogradmeta.as_ref() {
            meta.output_nr
        } else {
            0
        }
    }

    pub fn is_leaf(&self) -> bool {
        if let Some(meta) = TensorHook::get_autograd_meta(self) {
            meta.grad_fn_.as_ref().is_none()
        } else {
            true
        }
    }

    pub fn tensor_data(&self) -> Self {
        TensorHook::tensor_data(self)
    }

    pub fn get_tensor_impl(&self) -> *mut TensorImpl {
        let t = self._impl.as_ptr();
        t
        // todo!()
        // self._impl.borrow_mut()
    }
}

#[derive(Clone, Debug)]
pub struct Edge {
    pub function: Option<Rc<RefCell<Node>>>,
    pub input_nr: usize,
}

impl Edge {
    pub fn empty() -> Edge {
        Edge {
            function: None,
            input_nr: 0,
        }
    }
    pub fn new(function: Rc<RefCell<Node>>, input_nr: usize) -> Edge {
        let n = Rc::into_raw(function);
        let q = Some(unsafe { Rc::from_raw(n) });

        Edge {
            function: q,
            input_nr,
        }
    }
}
#[cfg(test)]
mod test {
    use crate::tensor::*;
    #[test]
    fn test_zeros() {
        let a = Tensor::zeros(&[2, 2]);
        println!("{}", a._impl.borrow().data);
    }
}
