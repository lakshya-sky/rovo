use super::tensor_ops;
use crate::aten::native;
use crate::c10::{
    type_meta_to_scalar_type, Device, Layout, MemoryFormat, ScalarType, Storage, TypeMeta,
};
use crate::core::Generator;
use crate::ops::*;
use crate::tensor::*;
use crate::util_autograd::TensorHook;
use std::cell::RefCell;
use std::rc::Rc;
use std::{ffi::c_void, ptr::NonNull};

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
#[derive(Clone, Default)]
pub struct NewTensor {
    pub _impl: Rc<RefCell<NewTensorImpl>>,
}

unsafe impl Send for NewTensor {}
unsafe impl Sync for NewTensor {}

// impl Default for NewTensor {
//     fn default() -> Self {
//         let impl_ = SINGLETON.get_or_init(|| Self::from_impl(NewTensorImpl::undefined_instance()));
//         impl_.clone()
//     }
// }
impl NewTensor {
    pub fn from_impl(_impl: NewTensorImpl) -> Self {
        Self {
            _impl: Rc::new(RefCell::new(_impl)),
        }
    }
    pub fn new(other: &Self) -> Self {
        Self {
            _impl: other._impl.clone(),
        }
    }
    pub fn get_unsafe_tensor_impl(&self) -> &mut NewTensorImpl {
        let t = self._impl.as_ptr();
        unsafe { &mut *t }
    }

    pub fn move_tensor(&mut self, other: NewTensor) {
        let impl_ = Rc::try_unwrap(other._impl);
        if let Ok(impl_) = impl_ {
            let tensor_impl = impl_.into_inner();
            self._impl.replace(tensor_impl);
            // self._impl = Rc::new(RefCell::new(tensor_impl));
        }
        else{
            todo!();
        }
        // self._impl = other._impl
    }

    pub fn is_same(&self, other: &Self) -> bool {
        self._impl.as_ptr() == other._impl.as_ptr()
    }

    pub fn storage(&self) -> &Storage {
        self.get_unsafe_tensor_impl().storage()
    }

    pub fn fill_(&self, value: f32) -> &Self {
        crate::aten::native::fill_(self, value)
    }
    pub fn sizes(&self) -> &[usize] {
        self.get_unsafe_tensor_impl().sizes()
    }
    pub fn strides(&self) -> &[usize] {
        self.get_unsafe_tensor_impl().strides()
    }
    pub fn storage_offset(&self) -> usize {
        self.get_unsafe_tensor_impl().storage_offset()
    }
    pub fn dim(&self) -> i64 {
        self.get_unsafe_tensor_impl().dim()
    }
    pub fn numel(&self) -> usize {
        self.get_unsafe_tensor_impl().numel()
    }
    pub fn size(&self, d: usize) -> usize {
        self.get_unsafe_tensor_impl().size(d)
    }
    pub fn resize(
        &self,
        size: &[usize],
        optional_memory_format: Option<crate::c10::MemoryFormat>,
    ) -> &NewTensor {
        native::resize(self, size, optional_memory_format)
    }
    pub fn element_size(&self) -> usize {
        self.get_unsafe_tensor_impl().itemsize()
    }
    pub fn defined(&self) -> bool {
        self.get_unsafe_tensor_impl().defined()
    }

    pub fn grad(&self) -> Option<Self> {
        self._impl.borrow().grad()
    }

    pub fn set_grad(&mut self, other: Self) {
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

    pub fn tensor_data(&self) -> Self {
        TensorHook::tensor_data(self)
    }

    pub fn is_contiguous(&self) -> bool {
        self.is_contiguous_(MemoryFormat::Contiguous)
    }

    pub fn is_contiguous_(&self, memory_format: MemoryFormat) -> bool {
        self.get_unsafe_tensor_impl().is_contiguous_(memory_format)
    }
    pub fn data_ptr(&self) -> NonNull<c_void> {
        self.get_unsafe_tensor_impl().data()
    }

    pub fn scalar_type(&self) -> ScalarType {
        type_meta_to_scalar_type(self.get_unsafe_tensor_impl().dtype())
    }
    pub fn dtype(&self) -> &TypeMeta {
        self.get_unsafe_tensor_impl().dtype()
    }
    pub fn device(&self) -> Device {
        self.get_unsafe_tensor_impl().device()
    }
    pub fn layout(&self) -> Layout {
        self.get_unsafe_tensor_impl().layout()
    }

    pub fn uniform(&self, from: f64, to: f64) {
        crate::aten::native::distribution_templates::uniform_impl_(self, from, to, None);
    }

    pub fn uniform_with_gen(&self, from: f64, to: f64, gen: Option<Generator>) {
        crate::aten::native::distribution_templates::uniform_impl_(self, from, to, gen);
    }

    pub fn randn(_dims: &[usize]) -> Self {
        // Self::from_impl(NewTensorImpl::randn(dims))
        todo!()
    }

    pub fn t(&self) -> Self {
        tensor_ops::t(self)
    }

    pub fn matmul(&self, _other: &NewTensor, _consume: bool) -> NewTensor {
        // println!("Matmul Shapes: {:?} and {:?}", self.sizes(), other.sizes());
        // super::linear_algebra::matmul(self, other, consume)
        todo!()
    }

    pub fn dot(&self, _other: &NewTensor) -> NewTensor {
        // super::linear_algebra::dot(self, other)
        todo!()
    }

    pub fn mm(&self, _other: &NewTensor, _consume: bool) -> NewTensor {
        // super::tensor_ops::mm(self, other, consume)
        todo!()
    }

    pub fn add_(&self, other: &NewTensor, _scalar: f64) {
        native::add_out(self, self, other);
    }

    pub fn sum(&self) -> Self {
        tensor_ops::sum(self, None, false)
    }

    pub fn mean(&self) -> Self {
        tensor_ops::mean(self)
    }

    pub fn sum_dim(&self, dims: &[usize], keep_dim: bool) -> NewTensor {
        tensor_ops::sum(self, Some(dims), keep_dim)
    }

    pub fn expand(&self, _size: &[usize]) -> NewTensor {
        todo!()
    }

    pub fn detach_(&mut self) -> &Self {
        let autograd_meta = crate::util_autograd::TensorHook::materialize_autograd_meta(self);
        autograd_meta.set_requires_grad(false);
        autograd_meta.grad_fn_reset();
        autograd_meta.output_nr = 0;
        self
    }

    pub fn zero_(&mut self) -> &Self {
        self.fill_(0.0)
    }

    pub fn mul_(&mut self, other: &NewTensor) {
        native::mul_out(self, self, other);
    }

    pub fn div_(&mut self, other: &NewTensor) {
        native::div_out(self, self, other);
    }

    pub fn options(&self) -> TensorOptions {
        let options = TensorOptions::default();
        options
            .set_dtype(*self.dtype())
            .set_device(self.device())
            .set_layout(self.layout());
        options
    }
}
