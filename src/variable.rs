use crate::autogradmeta::*;
use crate::ndarry_ext::*;
use std::cell::RefCell;
use std::rc::Rc;

pub struct TensorImpl {
    pub data: NdArray<f64>,
    pub autogradmeta: Option<AutogradMeta>,
}

impl TensorImpl {
    pub fn from_scalar(shape: &[usize], scalar: f64) -> TensorImpl {
        TensorImpl {
            data: NdArray::<f64>::from_elem(shape, scalar),
            autogradmeta: None,
        }
    }
    pub fn ones(shape: &[usize]) -> TensorImpl {
        TensorImpl {
            data: NdArray::<f64>::ones(shape),
            autogradmeta: None,
        }
    }
    pub fn zeros(shape: &[usize]) -> TensorImpl {
        TensorImpl {
            data: NdArray::<f64>::zeros(shape),
            autogradmeta: None,
        }
    }

    pub fn grad(&self) -> Option<Rc<Tensor>> {
        // if self.autogradmeta.is_none() {
        //     self.autogradmeta = Some(AutogradMetaFactory::make());
        // }
        self.autogradmeta.as_ref().unwrap().grad()
    }

    pub fn set_grad(&mut self, grad: Tensor) {
        if self.autogradmeta.is_none() {
            self.autogradmeta = Some(AutogradMetaFactory::make());
        }
        self.autogradmeta.as_mut().unwrap().set_grad(grad)
    }
}
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

    pub fn from_scalar(shape: &[usize], scalar: f64) -> Tensor {
        Tensor {
            _impl: Rc::new(RefCell::new(TensorImpl::from_scalar(shape, scalar))),
        }
    }

    pub fn ones(shape: &[usize]) -> Tensor {
        Tensor {
            _impl: Rc::new(RefCell::new(TensorImpl::ones(shape))),
        }
    }

    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor {
            _impl: Rc::new(RefCell::new(TensorImpl::zeros(shape))),
        }
    }

    pub fn grad(&mut self) -> Option<Rc<Tensor>> {
        self._impl.borrow().grad()
    }

    pub fn set_grad(&mut self, other: Tensor) {
        self._impl.borrow_mut().set_grad(other);
    }
}

#[cfg(test)]
mod test {
    use crate::variable::*;
    #[test]
    fn test_zeros() {
        let a = Tensor::zeros(&[2, 2]);
        println!("{}", a._impl.borrow().data);
    }
}
