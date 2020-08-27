use super::tensor_util;
use crate::autograd::*;
use crate::ndarry_ext::*;
use crate::ops::*;
use crate::tensor::*;
use ndarray_rand::{rand_distr, RandomExt};
use std::cell::RefCell;
use std::rc::Rc;
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
        if let Some(meta) = self.autogradmeta.as_ref() {
            meta.requires_grad()
        } else {
            false
        }
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
