use crate::aten::native::*;
use crate::autograd::SavedTensor;
use crate::ops::*;
use crate::tensor::*;
use crate::util_autograd;
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

impl Add<Self> for NewTensor {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl Add<Self> for &NewTensor {
    type Output = NewTensor;
    fn add(self, rhs: Self) -> Self::Output {
        let mut grad_fn: Option<Rc<RefCell<Node>>> = None;
        if util_autograd::compute_requires_grad(&[self, rhs]) {
            grad_fn = Some(Rc::new(RefCell::new(Node::new(AddBackwardTensors {
                next_edges: None,
                input_metadata_: smallvec::smallvec![],
            }))));
            grad_fn
                .as_mut()
                .unwrap()
                .borrow_mut()
                .set_next_edges(util_autograd::collect_next_edges(&[self, rhs]));
        }
        let result = add(self, rhs, 1.0);

        if grad_fn.is_some() {
            util_autograd::set_history(&result, grad_fn.unwrap());
        }
        result
    }
}

impl Add<f64> for &NewTensor {
    type Output = NewTensor;
    fn add(self, _rhs: f64) -> Self::Output {
        let mut grad_fn: Option<Rc<RefCell<Node>>> = None;
        if util_autograd::compute_requires_grad(&[&self]) {
            let mut _grad_fn = AddBackwardScalar {
                next_edges: None,
                input_metadata_: smallvec::smallvec![],
            };
            _grad_fn.set_next_edges(util_autograd::collect_next_edges(&[&self]));
            grad_fn = Some(Rc::new(RefCell::new(Node::new(_grad_fn))));
        }

        // let result = &self._impl.borrow().data + rhs;
        // let _impl = NewTensorImpl {
        //     data: result,
        //     autogradmeta: None,
        //     version_counter: TensorVersion::new(),
        // };
        // let result = NewTensor::from_impl(_impl);
        // if grad_fn.is_some() {
        //     util_autograd::set_history(&result, grad_fn.unwrap());
        // }
        // result
        todo!();
    }
}

impl Add<f64> for NewTensor {
    type Output = NewTensor;
    fn add(self, rhs: f64) -> Self::Output {
        let result = &self + rhs;
        result
    }
}

impl Mul<Self> for &NewTensor {
    type Output = NewTensor;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut grad_fn: Option<Rc<RefCell<Node>>> = None;
        if util_autograd::compute_requires_grad(&[self, rhs]) {
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
        let result = mul(self, rhs);

        if grad_fn.is_some() {
            util_autograd::set_history(&result, grad_fn.unwrap());
        }
        result
    }
}

impl Mul<f64> for &NewTensor {
    type Output = NewTensor;
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
        // let result = &self._impl.borrow().data * rhs;
        // let _impl = NewTensorImpl {
        //     data: result,
        //     autogradmeta: None,
        //     version_counter: TensorVersion::new(),
        // };
        // let result = NewTensor::from_impl(_impl);
        // if grad_fn.is_some() {
        //     util_autograd::set_history(&result, grad_fn.unwrap());
        // }
        // result
        todo!();
    }
}

impl Mul<f64> for NewTensor {
    type Output = NewTensor;
    fn mul(self, rhs: f64) -> Self::Output {
        let result = &self * rhs;
        result
    }
}

impl Mul<&Self> for NewTensor {
    type Output = NewTensor;
    fn mul(self, rhs: &Self) -> Self::Output {
        // let result = &self._impl.borrow().data * &rhs._impl.borrow().data;
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

        // let _impl = NewTensorImpl {
        //     data: result,
        //     autogradmeta: None,
        //     version_counter: TensorVersion::new(),
        // };
        // let result = NewTensor::from_impl(_impl);
        // if grad_fn.is_some() {
        //     util_autograd::set_history(&result, grad_fn.unwrap());
        // }
        // result
        todo!();
    }
}

impl Sub<Self> for &NewTensor {
    type Output = NewTensor;
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
        let result = sub(self, rhs, 1.0);

        if grad_fn.is_some() {
            util_autograd::set_history(&result, grad_fn.unwrap());
        }
        result
    }
}

impl Div<Self> for &NewTensor {
    type Output = NewTensor;
    fn div(self, rhs: Self) -> Self::Output {
        let mut grad_fn: Option<Rc<RefCell<Node>>> = None;
        if util_autograd::compute_requires_grad(&[self, rhs]) {
            let mut _grad_fn = DivBackwardTensors {
                next_edges: None,
                input_metadata_: smallvec::smallvec![],
                _self: None,
                other: None,
            };
            _grad_fn.set_next_edges(util_autograd::collect_next_edges(&[self, rhs]));
            _grad_fn._self = Some(SavedTensor::new(self, false));
            _grad_fn.other = Some(SavedTensor::new(rhs, false));
            grad_fn = Some(Rc::new(RefCell::new(Node::new(_grad_fn))));
        }
        let result = div(self, rhs);
        if grad_fn.is_some() {
            util_autograd::set_history(&result, grad_fn.unwrap());
        }
        result
    }
}

impl Div<Self> for NewTensor {
    type Output = NewTensor;
    fn div(self, rhs: Self) -> Self::Output {
        &self / &rhs
    }
}

impl Neg for &NewTensor {
    type Output = NewTensor;
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
        // let result = -&self._impl.borrow().data;
        // let _impl = NewTensorImpl {
        //     data: result,
        //     autogradmeta: None,
        //     version_counter: TensorVersion::new(),
        // };
        // let result = NewTensor::from_impl(_impl);
        // if grad_fn.is_some() {
        //     util_autograd::set_history(&result, grad_fn.unwrap());
        // }
        // result

        todo!()
    }
}

pub fn t(_self: &NewTensor) -> NewTensor {
    let mut grad_fn: Option<Rc<RefCell<Node>>> = None;
    if util_autograd::compute_requires_grad(&[_self]) {
        let mut _grad_fn = TBackward {
            next_edges: None,
            input_metadata_: smallvec::smallvec![],
        };
        _grad_fn.set_next_edges(util_autograd::collect_next_edges(&[_self]));
        grad_fn = Some(Rc::new(RefCell::new(Node::new(_grad_fn))));
    }
    // let _impl = _self._impl.borrow().t();
    // _impl.bump_version();
    // let result = NewTensor::from_impl(_impl);
    // if grad_fn.is_some() {
    //     util_autograd::set_history(&result, grad_fn.unwrap());
    // }
    // result
    todo!()
}

pub fn mm(_mat1: &NewTensor, _mat2: &NewTensor, _consume: bool) -> NewTensor {
    // let mut grad_fn: Option<Rc<RefCell<Node>>> = None;
    // if util_autograd::compute_requires_grad(&[mat1, mat2]) {
    //     let mut _grad_fn = MmBackward {
    //         next_edges: None,
    //         input_metadata_: smallvec::smallvec![],
    //         self_: None,
    //         mat2_: None,
    //         mat2_sizes: vec![],
    //     };
    //     _grad_fn.set_next_edges(util_autograd::collect_next_edges(&[mat1, mat2]));
    //     _grad_fn.self_ = Some(SavedTensor::new(mat1, false));
    //     if consume {
    //         _grad_fn.mat2_ = Some(SavedTensor::new_consume(mat2.clone(), false));
    //     } else {
    //         _grad_fn.mat2_ = Some(SavedTensor::new(&mat2, false));
    //     }
    //     _grad_fn.mat2_sizes = mat2._impl.borrow().data.shape().to_vec();

    //     grad_fn = Some(Rc::new(RefCell::new(Node::new(_grad_fn))));
    // }
    // let result = super::linear_algebra::mm(mat1, mat2);

    // // let result = NewTensor::from_impl(_impl);
    // if grad_fn.is_some() {
    //     util_autograd::set_history(&result, grad_fn.unwrap());
    // }
    // result
    todo!()
}

pub fn mean(self_: &NewTensor) -> NewTensor {
    let mut grad_fn: Option<Rc<RefCell<Node>>> = None;
    if util_autograd::compute_requires_grad(&[self_]) {
        let mut _grad_fn = MeanBackward {
            next_edges: None,
            input_metadata_: smallvec::smallvec![],
            self_numel: self_.numel(),
            self_sizes: self_.sizes().to_vec(),
        };
        _grad_fn.set_next_edges(util_autograd::collect_next_edges(&[self_]));
        grad_fn = Some(Rc::new(RefCell::new(Node::new(_grad_fn))));
    }
    // let result = super::linear_algebra::mean(self_);
    // if grad_fn.is_some() {
    //     util_autograd::set_history(&self_, grad_fn.unwrap());
    // }
    // result
    todo!()
}

pub fn sum(self_: &NewTensor, _dims: Option<&[usize]>, _keep_dim: bool) -> NewTensor {
    let mut grad_fn: Option<Rc<RefCell<Node>>> = None;
    if util_autograd::compute_requires_grad(&[self_]) {
        let mut _grad_fn = AddBackwardTensors {
            next_edges: None,
            input_metadata_: smallvec::smallvec![],
        };
        _grad_fn.set_next_edges(util_autograd::collect_next_edges(&[self_]));
        grad_fn = Some(Rc::new(RefCell::new(Node::new(_grad_fn))));
    }

    // let result = super::linear_algebra::sum(self_, dims, keep_dim);

    // if grad_fn.is_some() {
    //     util_autograd::set_history(&result, grad_fn.unwrap());
    // }

    // result
    todo!()
}

pub fn sigmoid(_tensor: &NewTensor) -> NewTensor {
    // let data = tensor.get_tensor_impl().data.clone();
    // let data = data.mapv(f64::exp);
    // // e^x / 1 + e^x instead of 1/1+e^-x
    // let data = data.clone() / (1.0 + data);
    // let result = NewTensor::from_impl(NewTensorImpl::new_from_array(data, false));

    // let mut grad_fn: Option<Rc<RefCell<Node>>> = None;
    // if util_autograd::compute_requires_grad(&[tensor]) {
    //     let mut _grad_fn = SigmoidBackward {
    //         next_edges: None,
    //         input_metadata_: smallvec::smallvec![],
    //         result_: None,
    //     };
    //     _grad_fn.set_next_edges(util_autograd::collect_next_edges(&[tensor]));
    //     _grad_fn.result_ = Some(SavedTensor::new(&result, true));
    //     grad_fn = Some(Rc::new(RefCell::new(Node::new(_grad_fn))));
    // }

    // if grad_fn.is_some() {
    //     util_autograd::set_history(&result, grad_fn.unwrap());
    // }
    // result
    todo!()
}

pub fn binary_cross_entropy(
    _input: &NewTensor,
    _target: &NewTensor,
    _weight: Option<&NewTensor>,
    _reduction: super::loss::Reduction,
) -> NewTensor {
    // let mut grad_fn: Option<Rc<RefCell<Node>>> = None;
    // if util_autograd::compute_requires_grad(&[input]) {
    //     let mut _grad_fn = BinaryCrossEntropyBackward::default();
    //     _grad_fn.set_next_edges(util_autograd::collect_next_edges(&[input]));
    //     _grad_fn.self_ = Some(SavedTensor::new(input, false));
    //     _grad_fn.target_ = Some(SavedTensor::new(target, false));
    //     if let Some(weight) = weight {
    //         _grad_fn.weight_ = Some(SavedTensor::new(weight, false));
    //     }
    //     _grad_fn.reduction = reduction;
    //     grad_fn = Some(Rc::new(RefCell::new(Node::new(_grad_fn))));
    // }
    // let result = loss::binary_cross_entropy(input, target, weight, reduction);
    // if grad_fn.is_some() {
    //     util_autograd::set_history(&result, grad_fn.unwrap());
    // }
    // result
    todo!()
}
