use crate::autograd::*;
use crate::ndarry_ext::*;
use crate::ops::*;
use crate::tensor::*;
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
        let mut impl_ = TensorImpl {
            data: NdArray::<f64>::from_elem(shape, scalar),
            autogradmeta: None,
            version_counter: TensorVersion::new(),
        };

        if requires_grad {
            impl_.set_autograd_meta(Some(AutogradMeta::new_without_edge(&impl_, requires_grad)));
        }
        impl_
    }
    pub fn ones(shape: &[usize]) -> TensorImpl {
        TensorImpl {
            data: NdArray::<f64>::ones(shape),
            autogradmeta: None,
            version_counter: TensorVersion::new(),
        }
    }
    pub fn zeros(shape: &[usize]) -> TensorImpl {
        TensorImpl {
            data: NdArray::<f64>::zeros(shape),
            autogradmeta: None,
            version_counter: TensorVersion::new(),
        }
    }

    pub fn grad(&self) -> Option<Rc<Tensor>> {
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
}
