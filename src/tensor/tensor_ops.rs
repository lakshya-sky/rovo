use crate::autograd::SavedTensor;
use crate::ops::*;
use crate::tensor::*;
use crate::util_autograd;
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

impl Add<Self> for Tensor {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut grad_fn: Option<Rc<RefCell<Node>>> = None;
        if util_autograd::compute_requires_grad(&[&self, &rhs]) {
            grad_fn = Some(Rc::new(RefCell::new(Node::new(AddBackwardTensors {
                next_edges: None,
                input_metadata_: smallvec::smallvec![],
            }))));
            grad_fn
                .as_mut()
                .unwrap()
                .borrow_mut()
                .set_next_edges(util_autograd::collect_next_edges(&[&self, &rhs]));
        }
        let result = &self._impl.borrow().data + &rhs._impl.borrow().data;
        let _impl = TensorImpl {
            data: result,
            autogradmeta: None,
            version_counter: TensorVersion::new(),
        };
        let result = Tensor::from_impl(_impl);
        if grad_fn.is_some() {
            util_autograd::set_history(&result, grad_fn.unwrap());
        }
        result
    }
}

impl Add<Self> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: Self) -> Self::Output {
        let mut grad_fn: Option<Rc<RefCell<Node>>> = None;
        if util_autograd::compute_requires_grad(&[&self, &rhs]) {
            grad_fn = Some(Rc::new(RefCell::new(Node::new(AddBackwardTensors {
                next_edges: None,
                input_metadata_: smallvec::smallvec![],
            }))));
            grad_fn
                .as_mut()
                .unwrap()
                .borrow_mut()
                .set_next_edges(util_autograd::collect_next_edges(&[&self, &rhs]));
        }
        let result = &self._impl.borrow().data + &rhs._impl.borrow().data;
        let _impl = TensorImpl {
            data: result,
            autogradmeta: None,
            version_counter: TensorVersion::new(),
        };
        let result = Tensor::from_impl(_impl);
        if grad_fn.is_some() {
            util_autograd::set_history(&result, grad_fn.unwrap());
        }
        result
    }
}

impl Add<f64> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: f64) -> Self::Output {
        let mut grad_fn: Option<Rc<RefCell<Node>>> = None;
        if util_autograd::compute_requires_grad(&[&self]) {
            let mut _grad_fn = AddBackwardScalar {
                next_edges: None,
                input_metadata_: smallvec::smallvec![],
            };
            _grad_fn.set_next_edges(util_autograd::collect_next_edges(&[&self]));
            grad_fn = Some(Rc::new(RefCell::new(Node::new(_grad_fn))));
        }

        let result = &self._impl.borrow().data + rhs;
        let _impl = TensorImpl {
            data: result,
            autogradmeta: None,
            version_counter: TensorVersion::new(),
        };
        let result = Tensor::from_impl(_impl);
        if grad_fn.is_some() {
            util_autograd::set_history(&result, grad_fn.unwrap());
        }
        result
    }
}

impl Add<f64> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: f64) -> Self::Output {
        let result = &self + rhs;
        result
    }
}

impl Mul<Self> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut grad_fn: Option<Rc<RefCell<Node>>> = None;
        if util_autograd::compute_requires_grad(&[&self, &rhs]) {
            let mut _grad_fn = MulBackwardTensors {
                next_edges: None,
                input_metadata_: smallvec::smallvec![],
                _self: None,
                other: None,
            };
            _grad_fn.set_next_edges(util_autograd::collect_next_edges(&[&self, &rhs]));
            _grad_fn._self = Some(SavedTensor::new(self, false));
            _grad_fn.other = Some(SavedTensor::new(rhs, false));
            grad_fn = Some(Rc::new(RefCell::new(Node::new(_grad_fn))));
        }
        let result = &self._impl.borrow().data * &rhs._impl.borrow().data;
        let _impl = TensorImpl {
            data: result,
            autogradmeta: None,
            version_counter: TensorVersion::new(),
        };
        let result = Tensor::from_impl(_impl);
        if grad_fn.is_some() {
            util_autograd::set_history(&result, grad_fn.unwrap());
        }
        result
    }
}

impl Mul<f64> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: f64) -> Self::Output {
        let mut grad_fn: Option<Rc<RefCell<Node>>> = None;
        if util_autograd::compute_requires_grad(&[&self]) {
            let mut _grad_fn = MulBackwardScalar {
                next_edges: None,
                input_metadata_: smallvec::smallvec![],
                _self: None,
                other: rhs,
            };
            _grad_fn.set_next_edges(util_autograd::collect_next_edges(&[&self]));
            _grad_fn._self = Some(SavedTensor::new(self, false));
            grad_fn = Some(Rc::new(RefCell::new(Node::new(_grad_fn))));
        }
        let result = &self._impl.borrow().data * rhs;
        let _impl = TensorImpl {
            data: result,
            autogradmeta: None,
            version_counter: TensorVersion::new(),
        };
        let result = Tensor::from_impl(_impl);
        if grad_fn.is_some() {
            util_autograd::set_history(&result, grad_fn.unwrap());
        }
        result
    }
}

impl Mul<f64> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: f64) -> Self::Output {
        let result = &self * rhs;
        result
    }
}

impl Mul<&Self> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Self) -> Self::Output {
        let result = &self._impl.borrow().data * &rhs._impl.borrow().data;
        let mut grad_fn: Option<Rc<RefCell<Node>>> = None;
        if util_autograd::compute_requires_grad(&[&self, &rhs]) {
            let mut _grad_fn = MulBackwardTensors {
                next_edges: None,
                input_metadata_: smallvec::smallvec![],
                _self: None,
                other: None,
            };
            _grad_fn.set_next_edges(util_autograd::collect_next_edges(&[&self, &rhs]));
            _grad_fn._self = Some(SavedTensor::new_consume(self, false));
            _grad_fn.other = Some(SavedTensor::new(rhs, false));
            grad_fn = Some(Rc::new(RefCell::new(Node::new(_grad_fn))));
        }

        let _impl = TensorImpl {
            data: result,
            autogradmeta: None,
            version_counter: TensorVersion::new(),
        };
        let result = Tensor::from_impl(_impl);
        if grad_fn.is_some() {
            util_autograd::set_history(&result, grad_fn.unwrap());
        }
        result
    }
}

impl Sub<Self> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut grad_fn: Option<Rc<RefCell<Node>>> = None;
        if util_autograd::compute_requires_grad(&[&self, &rhs]) {
            let mut _grad_fn = SubBackwardTensors {
                next_edges: None,
                input_metadata_: smallvec::smallvec![],
            };
            _grad_fn.set_next_edges(util_autograd::collect_next_edges(&[&self, &rhs]));
            grad_fn = Some(Rc::new(RefCell::new(Node::new(_grad_fn))));
        }
        let result = &self._impl.borrow().data - &rhs._impl.borrow().data;
        let _impl = TensorImpl {
            data: result,
            autogradmeta: None,
            version_counter: TensorVersion::new(),
        };
        let result = Tensor::from_impl(_impl);
        if grad_fn.is_some() {
            util_autograd::set_history(&result, grad_fn.unwrap());
        }
        result
    }
}

impl Div<Self> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: Self) -> Self::Output {
        let mut grad_fn: Option<Rc<RefCell<Node>>> = None;
        if util_autograd::compute_requires_grad(&[&self, &rhs]) {
            let mut _grad_fn = DivBackwardTensors {
                next_edges: None,
                input_metadata_: smallvec::smallvec![],
                _self: None,
                other: None,
            };
            _grad_fn.set_next_edges(util_autograd::collect_next_edges(&[&self, &rhs]));
            _grad_fn._self = Some(SavedTensor::new(self, false));
            _grad_fn.other = Some(SavedTensor::new(rhs, false));
            grad_fn = Some(Rc::new(RefCell::new(Node::new(_grad_fn))));
        }
        let result = &self._impl.borrow().data / &rhs._impl.borrow().data;
        let _impl = TensorImpl {
            data: result,
            autogradmeta: None,
            version_counter: TensorVersion::new(),
        };
        let result = Tensor::from_impl(_impl);
        if grad_fn.is_some() {
            util_autograd::set_history(&result, grad_fn.unwrap());
        }
        result
    }
}

impl Div<Self> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: Self) -> Self::Output {
        let result = &self._impl.borrow().data / &rhs._impl.borrow().data;
        let mut grad_fn: Option<Rc<RefCell<Node>>> = None;
        if util_autograd::compute_requires_grad(&[&self, &rhs]) {
            let mut _grad_fn = DivBackwardTensors {
                next_edges: None,
                input_metadata_: smallvec::smallvec![],
                _self: None,
                other: None,
            };
            _grad_fn.set_next_edges(util_autograd::collect_next_edges(&[&self, &rhs]));
            _grad_fn._self = Some(SavedTensor::new_consume(self, false));
            _grad_fn.other = Some(SavedTensor::new_consume(rhs, false));
            grad_fn = Some(Rc::new(RefCell::new(Node::new(_grad_fn))));
        }
        let _impl = TensorImpl {
            data: result,
            autogradmeta: None,
            version_counter: TensorVersion::new(),
        };
        let result = Tensor::from_impl(_impl);
        if grad_fn.is_some() {
            util_autograd::set_history(&result, grad_fn.unwrap());
        }
        result
    }
}

impl Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        let mut grad_fn: Option<Rc<RefCell<Node>>> = None;
        if util_autograd::compute_requires_grad(&[&self]) {
            let mut _grad_fn = NegBackward {
                next_edges: None,
                input_metadata_: smallvec::smallvec![],
            };
            _grad_fn.set_next_edges(util_autograd::collect_next_edges(&[&self]));
            grad_fn = Some(Rc::new(RefCell::new(Node::new(_grad_fn))));
        }
        let result = -&self._impl.borrow().data;
        let _impl = TensorImpl {
            data: result,
            autogradmeta: None,
            version_counter: TensorVersion::new(),
        };
        let result = Tensor::from_impl(_impl);
        if grad_fn.is_some() {
            util_autograd::set_history(&result, grad_fn.unwrap());
        }
        result
    }
}

pub fn t(_self: &Tensor) -> Tensor {
    let mut grad_fn: Option<Rc<RefCell<Node>>> = None;
    if util_autograd::compute_requires_grad(&[_self]) {
        let mut _grad_fn = TBackward {
            next_edges: None,
            input_metadata_: smallvec::smallvec![],
        };
        _grad_fn.set_next_edges(util_autograd::collect_next_edges(&[_self]));
        grad_fn = Some(Rc::new(RefCell::new(Node::new(_grad_fn))));
    }
    let _impl = _self._impl.borrow().t();
    _impl.bump_version();
    let result = Tensor::from_impl(_impl);
    if grad_fn.is_some() {
        util_autograd::set_history(&result, grad_fn.unwrap());
    }
    result
}

pub fn mm(mat1: &Tensor, mat2: &Tensor, consume: bool) -> Tensor {
    let mut grad_fn: Option<Rc<RefCell<Node>>> = None;
    if util_autograd::compute_requires_grad(&[mat1, mat2]) {
        let mut _grad_fn = MmBackward {
            next_edges: None,
            input_metadata_: smallvec::smallvec![],
            self_: None,
            mat2_: None,
            mat2_sizes: vec![],
        };
        _grad_fn.set_next_edges(util_autograd::collect_next_edges(&[mat1, mat2]));
        _grad_fn.self_ = Some(SavedTensor::new(mat1, false));
        if consume {
            _grad_fn.mat2_ = Some(SavedTensor::new_consume(mat2.clone(), false));
        } else {
            _grad_fn.mat2_ = Some(SavedTensor::new(&mat2, false));
        }
        _grad_fn.mat2_sizes = mat2._impl.borrow().data.shape().to_vec();

        grad_fn = Some(Rc::new(RefCell::new(Node::new(_grad_fn))));
    }
    let result = super::linear_algebra::mm(mat1, mat2);

    // let result = Tensor::from_impl(_impl);
    if grad_fn.is_some() {
        util_autograd::set_history(&result, grad_fn.unwrap());
    }
    result
}
