use super::tensor_ops;
use crate::aten::native;
use crate::autograd::*;
use crate::c10::MemoryFormat;
use crate::core::Generator;
use crate::ops::*;
use crate::tensor::*;
use crate::util_autograd::TensorHook;
use std::cell::RefCell;
use std::rc::Rc;
use std::{ffi::c_void, ptr::NonNull};
#[derive(Clone)]
pub struct Tensor {
    pub _impl: Rc<RefCell<TensorImpl>>,
}

unsafe impl Send for Tensor {}

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

    pub fn move_tensor(&self, other: Tensor) {
        let impl_ = Rc::try_unwrap(other._impl);
        if let Ok(impl_) = impl_ {
            let tensor_impl = impl_.into_inner();
            self._impl.replace(tensor_impl);
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

    // pub fn from_slice()->{

    // }

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

    pub fn grad(&self) -> Option<Rc<RefCell<Tensor>>> {
        self._impl.borrow().grad()
    }

    pub fn set_grad(&mut self, other: Tensor) {
        self._impl.borrow_mut().set_grad(other);
    }

    pub fn requires_grad(&self) -> bool {
        self._impl.borrow().requires_grad()
    }

    pub fn set_requires_grad(&self, requires_grad: bool) {
        self._impl.borrow_mut().set_requires_grad(requires_grad);
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

    pub fn dim(&self) -> i64 {
        self.get_tensor_impl().dim()
    }

    pub fn sizes(&self) -> &[usize] {
        self.get_tensor_impl().sizes()
    }
    pub fn tensor_data(&self) -> Self {
        TensorHook::tensor_data(self)
    }

    pub fn get_tensor_impl(&self) -> &mut TensorImpl {
        let t = self._impl.as_ptr();
        unsafe { &mut *t }
    }

    pub fn uniform(dims: &[usize], from: f64, to: f64) -> Tensor {
        Self::from_impl(TensorImpl::uniform(dims, from, to))
    }

    pub fn uniform_(&self, from: f64, to: f64) {
        crate::aten::native::distribution_templates::uniform_impl_(self, from, to, None);
    }

    pub fn uniform_with_gen(&self, from: f64, to: f64, gen: Option<Generator>) {
        crate::aten::native::distribution_templates::uniform_impl_(self, from, to, gen);
    }
    pub fn randn(dims: &[usize]) -> Tensor {
        Self::from_impl(TensorImpl::randn(dims))
    }

    pub fn t(&self) -> Tensor {
        tensor_ops::t(self)
    }

    pub fn matmul(&self, other: &Tensor, consume: bool) -> Tensor {
        // println!("Matmul Shapes: {:?} and {:?}", self.sizes(), other.sizes());
        super::linear_algebra::matmul(self, other, consume)
    }

    pub fn dot(&self, other: &Tensor) -> Tensor {
        super::linear_algebra::dot(self, other)
    }

    pub fn mm(&self, other: &Tensor, consume: bool) -> Tensor {
        super::tensor_ops::mm(self, other, consume)
    }

    pub fn add_(&self, other: &Tensor, scalar: f64) {
        let data =
            self.get_tensor_impl().data.clone() + other.get_tensor_impl().data.clone() * scalar;

        self.get_tensor_impl().data = data;
    }

    pub fn sum(&self) -> Self {
        tensor_ops::sum(self, None, false)
    }

    pub fn mean(&self) -> Self {
        tensor_ops::mean(self)
    }

    pub fn sum_dim(&self, dims: &[usize], keep_dim: bool) -> Tensor {
        tensor_ops::sum(self, Some(dims), keep_dim)
    }

    pub fn expand(&self, size: &[usize]) -> Tensor {
        Tensor::from_impl(TensorImpl::new_from_array(
            self.get_tensor_impl()
                .data
                .broadcast(size)
                .unwrap()
                .into_owned(),
            false,
        ))
    }

    pub fn detach_(&mut self) -> &Self {
        let autograd_meta = crate::util_autograd::TensorHook::materialize_autograd_meta(self);
        autograd_meta.set_requires_grad(false);
        autograd_meta.grad_fn_reset();
        autograd_meta.output_nr = 0;
        self
    }

    pub fn zero_(&mut self) {
        self.get_tensor_impl().data.fill(0.0);
    }

    pub fn mul_(&mut self, other: &Tensor) {
        let self_data = &self.get_tensor_impl().data;
        let other_data = &other.get_tensor_impl().data;
        self.get_tensor_impl().data = self_data * other_data;
    }

    pub fn div_(&mut self, other: &Tensor) {
        let self_data = &self.get_tensor_impl().data;
        let other_data = &other.get_tensor_impl().data;
        self.get_tensor_impl().data = self_data / other_data;
    }

    pub fn empty_like(other: &Tensor) -> Tensor {
        Tensor::from_impl(TensorImpl::empty_like(other.get_tensor_impl()))
    }

    pub fn empty(size: &[usize]) -> Tensor {
        Tensor::from_impl(TensorImpl::empty(size))
    }

    pub fn numel(&self) -> usize {
        self.get_tensor_impl().data.len()
    }

    pub fn size(&self, dim: i64) -> usize {
        let dim = maybe_wrap_dim(dim, self.dim(), false);
        self.sizes()[dim]
    }
}

#[derive(Clone)]
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
    pub fn new(function: Option<Rc<RefCell<Node>>>, input_nr: usize) -> Edge {
        // let n = Rc::into_raw(function);
        // let q = Some(unsafe { Rc::from_raw(n) });

        Edge { function, input_nr }
    }

    pub fn function(&self) -> Option<&Rc<RefCell<Node>>> {
        self.function.as_ref()
    }
}
#[derive(Clone)]
pub struct NewTensor {
    pub _impl: Rc<RefCell<NewTensorImpl>>,
}

unsafe impl Send for NewTensor {}

impl NewTensor {
    pub fn from_impl(_impl: NewTensorImpl) -> NewTensor {
        NewTensor {
            _impl: Rc::new(RefCell::new(_impl)),
        }
    }

    pub fn get_unsafe_tensor_impl(&self) -> &mut NewTensorImpl {
        let t = self._impl.as_ptr();
        unsafe { &mut *t }
    }

    pub fn fill_(&self, value: f32) -> &NewTensor {
        crate::aten::native::fill_(self, value)
    }
    pub fn sizes(&self) -> &[usize] {
        self.get_unsafe_tensor_impl().sizes()
    }
    pub fn numel(&self) -> usize {
        self.get_unsafe_tensor_impl().numel()
    }
    pub fn resize(
        &self,
        size: &[usize],
        optional_memory_format: Option<crate::c10::MemoryFormat>,
    ) -> &NewTensor {
        native::resize(self, size, optional_memory_format)
    }
    pub fn element_size(&self) -> usize {
        self.get_unsafe_tensor_impl().item_size()
    }
    pub fn defined(&self) -> bool {
        // Todo: Pytorch checks for if impl is null or not
        true
    }
    pub fn is_contiguous(&self, memory_format: MemoryFormat) -> bool {
        self.get_unsafe_tensor_impl().is_contiguous(memory_format)
    }
    pub fn data_ptr(&self) -> NonNull<c_void> {
        self.get_unsafe_tensor_impl().data()
    }
}
impl std::fmt::Debug for NewTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} \nSize: {:?}",
            self.get_unsafe_tensor_impl().dim(),
            self.get_unsafe_tensor_impl()
        )
    }
}
