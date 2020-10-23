use super::tensor_util;
use crate::autograd::*;
use crate::c10::{Device, MemoryFormat, Storage, TypeMeta};
use crate::ndarry_ext::*;
use crate::ops::*;
use crate::tensor::*;
use ndarray_rand::{rand_distr, RandomExt};
use std::cell::RefCell;
use std::rc::Rc;
use std::{ffi::c_void, ptr::NonNull};
#[derive(Default, Debug)]
pub struct VersionCounter {
    pub version: u32,
}

impl VersionCounter {
    pub fn new(version: u32) -> Self {
        VersionCounter { version }
    }
}

#[derive(Clone, Debug)]
pub struct TensorVersion {
    pub version_counter: Rc<RefCell<VersionCounter>>,
}

impl TensorVersion {
    pub fn new() -> Self {
        TensorVersion {
            version_counter: Rc::new(RefCell::new(VersionCounter::default())),
        }
    }

    pub fn new_with_version(version: u32) -> Self {
        Self {
            version_counter: Rc::new(RefCell::new(VersionCounter::new(version))),
        }
    }

    pub fn bump(&self) {
        let mut version = self.version_counter.borrow_mut();
        version.version += 1;
    }

    pub fn current_version(&self) -> u32 {
        self.version_counter.borrow().version
    }

    pub fn unique(&self) -> bool {
        1 == Rc::strong_count(&self.version_counter)
    }
}

pub struct TensorImpl {
    pub data: NdArray<f64>,
    pub autogradmeta: Option<AutogradMeta>,
    pub version_counter: TensorVersion,
}

impl TensorImpl {
    pub fn from_scalar(shape: &[usize], scalar: f64, requires_grad: bool) -> TensorImpl {
        //Todo: this is weird because its what pytorch does.
        TensorImpl::new_from_array(NdArray::<f64>::from_elem(shape, scalar), requires_grad)
    }

    pub fn ones(shape: &[usize]) -> TensorImpl {
        TensorImpl::new_from_array(NdArray::<f64>::ones(shape), false)
    }

    pub fn zeros(shape: &[usize]) -> TensorImpl {
        TensorImpl::new_from_array(NdArray::<f64>::zeros(shape), false)
    }

    pub fn new_from_array(data: NdArray<f64>, requires_grad: bool) -> Self {
        let mut _impl = Self {
            data,
            autogradmeta: None,
            version_counter: TensorVersion::new(),
        };

        if requires_grad {
            _impl.set_autograd_meta(Some(AutogradMeta::new_without_edge(&_impl, requires_grad)));
        }
        _impl
    }

    pub fn empty_like(other: &Self) -> Self {
        Self::empty(other.sizes())
    }

    pub fn empty(size: &[usize]) -> Self {
        Self::new_from_array(unsafe { ndarray::Array::uninitialized(size) }, false)
    }

    pub fn grad(&self) -> Option<Rc<RefCell<Tensor>>> {
        self.autogradmeta.as_ref().unwrap().grad()
    }

    pub fn set_grad(&mut self, grad: Tensor) {
        if self.autogradmeta.is_none() {
            self.autogradmeta = Some(AutogradMetaFactory::make());
        }
        self.autogradmeta.as_mut().unwrap().set_grad(grad)
    }

    pub fn requires_grad(&self) -> bool {
        let result;
        if let Some(meta) = self.autogradmeta.as_ref() {
            result = meta.requires_grad()
        } else {
            result = false
        }
        eprintln!("Requires Grad: {}", result);
        result
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        if self.autogradmeta.is_none() {
            self.autogradmeta = Some(AutogradMetaFactory::make());
        }
        self.autogradmeta
            .as_mut()
            .unwrap()
            .set_requires_grad(requires_grad);
    }

    pub fn get_autogradmeta(&mut self) -> Option<&mut AutogradMeta> {
        self.autogradmeta.as_mut()
    }

    pub fn set_autograd_meta(&mut self, t: Option<AutogradMeta>) {
        self.autogradmeta = t;
    }

    pub fn set_grad_fn(&mut self, t: Option<Rc<RefCell<Node>>>) {
        let meta = self.autogradmeta.as_mut().unwrap();
        meta.grad_fn_ = t;
    }

    pub fn set_output_nr(&mut self, output_nr: usize) {
        let meta = self.autogradmeta.as_mut().unwrap();
        meta.output_nr = output_nr;
    }

    pub fn shallow_copy(&self) -> TensorImpl {
        let data = self.data.clone();
        TensorImpl {
            data,
            autogradmeta: None,
            version_counter: TensorVersion::new(),
        }
    }

    pub fn version_counter(&self) -> &TensorVersion {
        &self.version_counter
    }

    pub fn unique_version(&self) -> bool {
        self.version_counter.unique()
    }

    pub fn set_version_counter(&mut self, version_counter: TensorVersion) {
        self.version_counter = version_counter;
    }

    pub fn uniform(shape: &[usize], from: f64, to: f64) -> TensorImpl {
        let uniform = NdArray::<f64>::random(shape, rand_distr::Uniform::new(from, to));
        TensorImpl::new_from_array(uniform, false)
    }

    pub fn randn(shape: &[usize]) -> TensorImpl {
        let uniform = NdArray::<f64>::random(shape, rand_distr::StandardNormal);
        TensorImpl::new_from_array(uniform, false)
    }

    pub fn t(&self) -> Self {
        self.transpose(0, if self.dim() < 2 { 0 } else { 1 })
    }

    pub fn transpose(&self, dim0: i64, dim1: i64) -> Self {
        let ndims = self.dim();
        let dim0 = tensor_util::maybe_wrap_dim(dim0, ndims, true);
        let dim1 = tensor_util::maybe_wrap_dim(dim1, ndims, true);
        let mut tr_data = self.data.clone();
        tr_data.swap_axes(dim0, dim1);
        Self::new_from_array(tr_data, false)
    }

    pub fn dim(&self) -> i64 {
        self.data.ndim() as i64
    }

    pub fn sizes(&self) -> &[usize] {
        self.data.shape()
    }
    pub fn bump_version(&self) {
        self.version_counter.bump()
    }
}

pub struct NewTensorImpl {
    storage: Storage,
    pub autogradmeta: Option<AutogradMeta>,
    pub version_counter: TensorVersion,
    sizes: smallvec::SmallVec<[usize; 5]>,
    strides: smallvec::SmallVec<[usize; 5]>,
    storage_offset: usize,
    numel: usize,
    data_type: TypeMeta,
    device_opt: Option<Device>,
    is_contiguous: bool,
}

impl NewTensorImpl {
    pub fn new(storage: Storage, data_type: TypeMeta, device_opt: Option<Device>) -> Self {
        let sizes = smallvec::smallvec![0];
        let storage_offset = 0;
        let strides = smallvec::smallvec![1];
        let numel = 0;
        let version_counter = TensorVersion::new();
        Self {
            storage,
            data_type,
            device_opt,
            sizes,
            strides,
            storage_offset,
            numel,
            version_counter,
            autogradmeta: None,
            is_contiguous: true,
        }
    }
    // pub fn from_scalar(shape: &[usize], scalar: f64, requires_grad: bool) -> TensorImpl {
    //     //Todo: this is weird because its what pytorch does.
    //     TensorImpl::new_from_array(NdArray::<f64>::from_elem(shape, scalar), requires_grad)
    // }

    // pub fn ones(shape: &[usize]) -> TensorImpl {
    //     TensorImpl::new_from_array(NdArray::<f64>::ones(shape), false)
    // }

    // pub fn zeros(shape: &[usize]) -> TensorImpl {
    //     TensorImpl::new_from_array(NdArray::<f64>::zeros(shape), false)
    // }

    // pub fn new_from_array(data: NdArray<f64>, requires_grad: bool) -> Self {
    //     let mut _impl = Self {
    //         data,
    //         autogradmeta: None,
    //         version_counter: TensorVersion::new(),
    //     };

    //     if requires_grad {
    //         _impl.set_autograd_meta(Some(AutogradMeta::new_without_edge(&_impl, requires_grad)));
    //     }
    //     _impl
    // }

    // pub fn empty_like(other: &Self) -> Self {
    //     Self::empty(other.sizes())
    // }

    // pub fn empty(size: &[usize]) -> Self {
    //     Self::new_from_array(unsafe { ndarray::Array::uninitialized(size) }, false)
    // }

    pub fn grad(&self) -> Option<Rc<RefCell<Tensor>>> {
        self.autogradmeta.as_ref().unwrap().grad()
    }

    pub fn set_grad(&mut self, grad: Tensor) {
        if self.autogradmeta.is_none() {
            self.autogradmeta = Some(AutogradMetaFactory::make());
        }
        self.autogradmeta.as_mut().unwrap().set_grad(grad)
    }

    pub fn requires_grad(&self) -> bool {
        let result;
        if let Some(meta) = self.autogradmeta.as_ref() {
            result = meta.requires_grad()
        } else {
            result = false
        }
        eprintln!("Requires Grad: {}", result);
        result
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        if self.autogradmeta.is_none() {
            self.autogradmeta = Some(AutogradMetaFactory::make());
        }
        self.autogradmeta
            .as_mut()
            .unwrap()
            .set_requires_grad(requires_grad);
    }

    pub fn get_autogradmeta(&mut self) -> Option<&mut AutogradMeta> {
        self.autogradmeta.as_mut()
    }

    pub fn set_autograd_meta(&mut self, t: Option<AutogradMeta>) {
        self.autogradmeta = t;
    }

    pub fn set_grad_fn(&mut self, t: Option<Rc<RefCell<Node>>>) {
        let meta = self.autogradmeta.as_mut().unwrap();
        meta.grad_fn_ = t;
    }

    pub fn set_output_nr(&mut self, output_nr: usize) {
        let meta = self.autogradmeta.as_mut().unwrap();
        meta.output_nr = output_nr;
    }

    pub fn version_counter(&self) -> &TensorVersion {
        &self.version_counter
    }

    pub fn unique_version(&self) -> bool {
        self.version_counter.unique()
    }

    pub fn set_version_counter(&mut self, version_counter: TensorVersion) {
        self.version_counter = version_counter;
    }

    pub fn dim(&self) -> i64 {
        self.sizes.len() as i64
    }

    pub fn sizes(&self) -> &[usize] {
        self.sizes.as_slice()
    }
    pub fn bump_version(&self) {
        self.version_counter.bump()
    }
    pub fn set_sizes_contiguous(&mut self, new_size: &[usize]) {
        let new_dim = new_size.len();
        self.sizes.resize(new_dim, 0);
        for i in 0..new_dim {
            self.sizes[i] = new_size[i];
        }
        self.refresh_numel();
    }
    fn refresh_numel(&mut self) {
        self.numel = self.compute_numel();
    }
    fn compute_numel(&self) -> usize {
        self.sizes.iter().product()
    }
    pub fn numel(&self) -> usize {
        self.numel
    }
    pub fn storage_offset(&self) -> usize {
        self.storage_offset
    }
    pub fn dtype(&self) -> &TypeMeta {
        &self.data_type
    }
    pub fn storage(&self) -> &Storage {
        &self.storage
    }
    pub fn is_contiguous(&self, _memory_format: MemoryFormat) -> bool {
        self.is_contiguous
    }
    pub fn item_size(&self) -> usize {
        self.data_type.itemsize()
    }
    pub fn data(&self) -> NonNull<c_void> {
        let d_ptr = self.storage.data::<c_void>();
        unsafe {
            let new_ptr = d_ptr
                .cast::<u8>()
                .as_ptr()
                .add(self.data_type.itemsize() * self.storage_offset);
            NonNull::new_unchecked(new_ptr as *mut c_void)
        }
    }
    fn print_impl(&self) -> String {
        let mut vec: Vec<f32> = vec![];
        let data = self.data().as_ptr();

        for i in 0..self.numel {
            unsafe {
                let el = data.add(i * self.data_type.itemsize()) as *const f32;
                vec.push(*el);
            };
        }
        format!("{:?}", vec.as_slice())
    }
}
impl std::fmt::Debug for NewTensorImpl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.print_impl())
    }
}
