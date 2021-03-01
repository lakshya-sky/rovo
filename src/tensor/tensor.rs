use aten::core::{Number, TensorAccessor};
use native::wrapped_scalar_tensor;

use super::tensor_ops;
use crate::aten::{self, native};
use crate::c10::{
    type_meta_to_scalar_type, Device, Layout, MemoryFormat, Scalar, ScalarType, Storage,
    TensorOptions, TypeMeta,
};
use crate::core::Generator;
use crate::ops::*;
use crate::tensor::*;
use crate::util_autograd::TensorHook;
use std::cell::RefCell;
use std::rc::Rc;
use std::{ffi::c_void, ptr::NonNull};

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
    pub fn new(function: Option<Rc<RefCell<Node>>>, input_nr: usize) -> Edge {
        // let n = Rc::into_raw(function);
        // let q = Some(unsafe { Rc::from_raw(n) });
        Edge { function, input_nr }
    }

    pub fn function(&self) -> Option<&Rc<RefCell<Node>>> {
        self.function.as_ref()
    }
    pub fn is_valid(&self) -> bool {
        self.function.is_some()
    }
}

#[derive(Clone, Default)]
pub struct Tensor {
    pub _impl: Rc<RefCell<TensorImpl>>,
}

unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

// impl Default for Tensor {
//     fn default() -> Self {
//         let impl_ = SINGLETON.get_or_init(|| Self::from_impl(TensorImpl::undefined_instance()));
//         impl_.clone()
//     }
// }
impl Tensor {
    pub fn from_impl(_impl: TensorImpl) -> Self {
        Self {
            _impl: Rc::new(RefCell::new(_impl)),
        }
    }
    pub fn new(other: &Self) -> Self {
        Self {
            _impl: other._impl.clone(),
        }
    }
    pub fn get_unsafe_tensor_impl(&self) -> &mut TensorImpl {
        let t = self._impl.as_ptr();
        unsafe { &mut *t }
    }

    pub fn move_tensor(&mut self, other: Tensor) {
        let impl_ = Rc::try_unwrap(other._impl);
        if let Ok(impl_) = impl_ {
            let tensor_impl = impl_.into_inner();
            self._impl.replace(tensor_impl);
        // self._impl = Rc::new(RefCell::new(tensor_impl));
        } else {
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

    pub fn fill_(&self, value: impl Into<Scalar>) {
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

    pub fn ndimension(&self) -> usize {
        self.dim() as usize
    }

    pub fn numel(&self) -> usize {
        self.get_unsafe_tensor_impl().numel()
    }
    pub fn size(&self, d: i64) -> usize {
        self.get_unsafe_tensor_impl().size(d)
    }
    pub fn stride(&self, d: usize) -> usize {
        self.get_unsafe_tensor_impl().stride(d)
    }
    pub fn resize_as_(&self, other: &Self) -> &Self {
        native::resize_as_(self, other, None)
    }

    pub fn resize(
        &self,
        size: &[usize],
        optional_memory_format: Option<crate::c10::MemoryFormat>,
    ) -> &Tensor {
        native::resize(self, size, optional_memory_format)
    }

    pub fn reshape(&self, shape: &[isize]) -> Self {
        native::reshape(&self, shape)
    }

    pub fn copy(&self, src: &Self, non_blocking: Option<bool>) -> &Self {
        native::copy_(self, src, non_blocking.unwrap_or(false))
    }

    pub fn as_strided(&self, size: &[usize], strides: &[usize]) -> Self {
        aten::as_strided(self, size, strides, None)
    }

    pub fn as_strided_(&self, size: &[usize], strides: &[usize]) -> &Self {
        aten::native::as_strided_(self, size, strides, None)
    }
    pub fn contiguous(&self) -> Self {
        aten::native::contiguous(self)
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
            p.grad_fn()
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

    pub fn is_non_overlapping_and_dense(&self) -> bool {
        self.get_unsafe_tensor_impl().is_non_overlapping_and_dense()
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

    pub fn data_ptr_casted<T>(&self) -> *mut T {
        self.get_unsafe_tensor_impl().data().as_ptr() as *mut T
    }

    pub fn accessor<T, N: Number>(&self, len: usize) -> TensorAccessor<T, N> {
        TensorAccessor::new(
            self.data_ptr_casted::<T>(),
            self.sizes().as_ptr(),
            self.strides().as_ptr(),
            len,
        )
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
    pub fn unsqueeze(&self, _i: usize) -> Self {
        todo!()
    }

    pub fn randn(_dims: &[usize]) -> Self {
        // Self::from_impl(TensorImpl::randn(dims))
        todo!()
    }

    pub fn t(&self) -> Self {
        tensor_ops::t(self)
    }

    pub fn transpose(&self, dim0: i64, dim1: i64) -> Self {
        tensor_ops::transpose(self, dim0, dim1)
    }

    pub fn transpose_(&self, dim0: i64, dim1: i64) -> &Self {
        tensor_ops::transpose_(self, dim0, dim1)
    }

    pub fn matmul(&self, other: &Tensor, consume: bool) -> Tensor {
        // println!("Matmul Shapes: {:?} and {:?}", self.sizes(), other.sizes());
        native::matmul(self, other, consume)
    }

    pub fn dot(&self, _other: &Tensor) -> Tensor {
        // super::tensor_ops::dot(self, other)
        todo!()
    }

    pub fn mm<T: AsRef<Tensor>>(&self, other: T, consume: bool) -> Tensor {
        super::tensor_ops::mm(self, other, consume)
    }

    pub fn add_(&self, other: &Tensor) {
        self.add_with_alpha_(other, 1.0);
    }

    pub fn add_scalar(&self, other: impl Into<Scalar>) {
        let other = wrapped_scalar_tensor(other.into());
        self.add_(&other);
    }

    pub fn add_with_alpha_(&self, other: &Tensor, scalar: impl Into<Scalar>) {
        native::add_out(self, self, other, scalar.into());
    }

    pub fn sum(&self) -> Self {
        tensor_ops::sum(self, None)
    }

    pub fn mean(&self) -> Self {
        tensor_ops::mean(self)
    }

    pub fn sum_dim(&self, dims: &[usize], keep_dim: bool) -> Tensor {
        tensor_ops::sum_dim_int_list(self, dims, keep_dim)
    }
    pub fn log_softmax(&self, dim: i64, dtype: Option<ScalarType>) -> Tensor {
        native::log_softmax(self, dim, dtype)
    }
    /// Implicit flag is false by default, and is only true for
    /// oprations which broadcasts tensor implicitly.
    pub fn expand(&self, size: &[usize], implicit: bool) -> Tensor {
        aten::native::expand(self, size, implicit)
    }

    pub fn detach_(&mut self) -> &Self {
        let autograd_meta = crate::util_autograd::TensorHook::materialize_autograd_meta(self);
        autograd_meta.set_requires_grad(false);
        autograd_meta.grad_fn_reset();
        autograd_meta.output_nr = 0;
        self
    }

    pub fn zero_(&mut self) {
        self.fill_(0.0)
    }

    pub fn mul_<A: AsRef<Self>>(&mut self, other: A) {
        native::mul_out(self, self, other.as_ref());
    }

    pub fn mul_scalar<S: Into<Scalar>>(&mut self, other: S) {
        self.mul_(wrapped_scalar_tensor(other.into()));
    }

    pub fn div_<A: AsRef<Self>>(&mut self, other: A) {
        native::div_out(self, self, other.as_ref());
    }

    pub fn div_scalar<S: Into<Scalar>>(&mut self, other: S) {
        self.div_(wrapped_scalar_tensor(other.into()));
    }

    pub fn view(&self, shape: &[usize]) -> Self {
        aten::native::view(self, shape)
    }

    pub fn options(&self) -> TensorOptions {
        let options = TensorOptions::default();
        options
            .set_dtype(*self.dtype())
            .set_device(self.device())
            .set_layout(self.layout());
        options
    }

    pub fn to_dtype(&self, dtype: ScalarType) -> Self {
        native::to_dtype(self, dtype, false, false, None)
    }

    /// Defaults:
    ///     channels_last_strides_exact_match: false
    pub fn suggest_memory_format(&self, _channels_last_strides_exact_match: bool) -> MemoryFormat {
        // Setting channels_last_strides_exact_match to true forces function to
        // check 0,1 - sized dimension strides.
        //   if !self.is_mkldnn() && !self.is_sparse() {
        //     if (impl_->is_strides_like_channels_last()) {
        //       if (!channels_last_strides_exact_match ||
        //           get_channels_last_strides_2d(sizes()) == strides()) {
        //         return at::MemoryFormat::ChannelsLast;
        //       }
        //     }
        //     else if (impl_->is_strides_like_channels_last_3d()) {
        //       if (!channels_last_strides_exact_match ||
        //           get_channels_last_strides_3d(sizes()) == strides()) {
        //         return at::MemoryFormat::ChannelsLast3d;
        //       }
        //     }
        //   }
        return MemoryFormat::Contiguous;
    }
}

impl AsRef<Self> for Tensor {
    fn as_ref(&self) -> &Self {
        self
    }
}

impl AsRef<Self> for &Tensor {
    fn as_ref(&self) -> &Self {
        self
    }
}
