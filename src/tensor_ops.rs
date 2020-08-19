use crate::autograd::SavedTensor;
use crate::ops::*;
use crate::tensor::*;
use crate::util;
use std::cell::RefCell;
use std::ops::{Add, Mul};
use std::rc::Rc;

impl Add<Self> for Tensor {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut grad_fn: Option<Rc<RefCell<Node>>> = None;
        if util::compute_requires_grad(&[&self, &rhs]) {
            grad_fn = Some(Rc::new(RefCell::new(Node::new(AddBackwardTensors {
                next_edges: None,
                input_metadata_: smallvec::smallvec![],
            }))));
            grad_fn
                .as_mut()
                .unwrap()
                .borrow_mut()
                .set_next_edges(util::collect_next_edges(&[&self, &rhs]));
        }
        let result = &self._impl.borrow().data + &rhs._impl.borrow().data;
        let _impl = TensorImpl {
            data: result,
            autogradmeta: None,
            version_counter: TensorVersion::new(),
        };
        let result = Tensor::from_impl(_impl);
        if grad_fn.is_some() {
            util::set_history(&result, grad_fn.unwrap());
        }
        result
    }
}

impl Add<Self> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: Self) -> Self::Output {
        let mut grad_fn: Option<Rc<RefCell<Node>>> = None;
        if util::compute_requires_grad(&[&self, &rhs]) {
            grad_fn = Some(Rc::new(RefCell::new(Node::new(
                AddBackwardTensors {
                    next_edges: None,
                    input_metadata_: smallvec::smallvec![],
                }),
            )));
            grad_fn
                .as_mut()
                .unwrap()
                .borrow_mut()
                .set_next_edges(util::collect_next_edges(&[&self, &rhs]));
        }
        let result = &self._impl.borrow().data + &rhs._impl.borrow().data;
        let _impl = TensorImpl {
            data: result,
            autogradmeta: None,
            version_counter: TensorVersion::new(),
        };
        let result = Tensor::from_impl(_impl);
        if grad_fn.is_some() {
            util::set_history(&result, grad_fn.unwrap());
        }
        result
    }
}

impl Mul<Self> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut grad_fn: Option<Rc<RefCell<Node>>> = None;
        if util::compute_requires_grad(&[&self, &rhs]) {
            let mut _grad_fn = MulBackwardTensors {
                next_edges: None,
                input_metadata_: smallvec::smallvec![],
                _self: None,
                other: None,
            };
            _grad_fn.set_next_edges(util::collect_next_edges(&[&self, &rhs]));
            _grad_fn._self = Some(SavedTensor::new(self, false));
            _grad_fn.other = Some(SavedTensor::new(rhs, false));
            grad_fn = Some(Rc::new(RefCell::new(Node::new(
                _grad_fn,
            ))));
        }
        let result = &self._impl.borrow().data * &rhs._impl.borrow().data;
        let _impl = TensorImpl {
            data: result,
            autogradmeta: None,
            version_counter: TensorVersion::new(),
        };
        let result = Tensor::from_impl(_impl);
        if grad_fn.is_some() {
            util::set_history(&result, grad_fn.unwrap());
        }
        result
    }
}

#[cfg(test)]
mod test {
    use crate::tensor::Tensor;

    #[test]
    fn test_add_op() {
        let t = Tensor::from_scalar(&[2, 2], 2.0, false);
        let x = Tensor::from_scalar(&[2, 2], 1.0, false);
        let res: Tensor = t + x;
        println!("{}", res._impl.borrow().data);
    }

    #[test]
    fn test_mul_op() {
        let t = Tensor::from_scalar(&[2, 2], 3.0, true);
        let x = Tensor::from_scalar(&[2, 2], 5.0, true);
        let res: Tensor = &t * &x;
        println!("{}", res._impl.borrow().data);
    }
}
