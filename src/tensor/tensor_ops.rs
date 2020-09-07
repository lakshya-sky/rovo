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
        println!("LHS: {:?}", &self._impl.borrow().data);
        println!("RHS: {:?}", &rhs._impl.borrow().data);
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

pub fn sum(self_: &Tensor, dims: Option<&[usize]>, keep_dim: bool) -> Tensor {
    let mut grad_fn: Option<Rc<RefCell<Node>>> = None;
    if util_autograd::compute_requires_grad(&[self_]) {
        let mut _grad_fn = AddBackwardTensors {
            next_edges: None,
            input_metadata_: smallvec::smallvec![],
        };
        _grad_fn.set_next_edges(util_autograd::collect_next_edges(&[self_]));
        grad_fn = Some(Rc::new(RefCell::new(Node::new(_grad_fn))));
    }

    let result = super::linear_algebra::sum(self_, dims, keep_dim);

    if grad_fn.is_some() {
        util_autograd::set_history(&result, grad_fn.unwrap());
    }

    result
}

pub fn sigmoid(tensor: &Tensor) -> Tensor {
    let data = tensor.get_tensor_impl().data.clone();
    let data = data.mapv(f64::exp);
    // e^x / 1 + e^x instead of 1/1+e^-x
    let data = data.clone() / (1.0 + data);
    let result = Tensor::from_impl(TensorImpl::new_from_array(data, false));

    let mut grad_fn: Option<Rc<RefCell<Node>>> = None;
    if util_autograd::compute_requires_grad(&[tensor]) {
        let mut _grad_fn = SigmoidBackward {
            next_edges: None,
            input_metadata_: smallvec::smallvec![],
            result_: None,
        };
        _grad_fn.set_next_edges(util_autograd::collect_next_edges(&[tensor]));
        _grad_fn.result_ = Some(SavedTensor::new(tensor, true));
        grad_fn = Some(Rc::new(RefCell::new(Node::new(_grad_fn))));
    }

    if grad_fn.is_some() {
        util_autograd::set_history(&result, grad_fn.unwrap());
    }
    result
}

pub fn binary_cross_entropy(
    self_: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    reduction: usize,
) -> Tensor {
    let mut grad_fn: Option<Rc<RefCell<Node>>> = None;
    if util_autograd::compute_requires_grad(&[self_]) {
        let mut _grad_fn = BinaryCrossEntropyBackward::default();
        _grad_fn.set_next_edges(util_autograd::collect_next_edges(&[self_]));
        _grad_fn.self_ = Some(SavedTensor::new(self_, true));
        _grad_fn.target_ = Some(SavedTensor::new(target, true));
        if let Some(weight) = weight {
            _grad_fn.weight_ = Some(SavedTensor::new(weight, true));
        }
        _grad_fn.reduction = reduction;
        grad_fn = Some(Rc::new(RefCell::new(Node::new(_grad_fn))));
    }

    if grad_fn.is_some() {
        util_autograd::set_history(self_, grad_fn.unwrap());
    }
    todo!()
}

pub fn binary_cross_entropy_backward(
    grad: &Tensor,
    input: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    reduction: usize,
) -> Tensor {
    let mut grad_input = Tensor::empty_like(input);
    binary_cross_entropy_backward_out(&mut grad_input, grad, input, target, weight, reduction);
    grad_input
}

pub fn binary_cross_entropy_backward_out(
    grad_input: &mut Tensor,
    _grad: &Tensor,
    input: &Tensor,
    target: &Tensor,
    _weight: Option<&Tensor>,
    _reduction: usize,
) {
    let iter = tensor_iterator::TensorIteratorConfig::default()
        .add_output(grad_input)
        .add_input(input)
        .add_input(target)
        .build();
    iter.for_each(|input_val, target_val| {
        let return_val = (target_val - 1.0) * ((1.0 - input_val).ln().max(-100.0))
            - (target_val * input_val.ln().max(-100.0));
        println!("Return value from closure {}", return_val);
        return_val
    });

    // todo!()
}

#[cfg(test)]
mod test {
    use crate::tensor::Tensor;
    #[test]
    fn bce_loss_test() {
        let input = Tensor::from_scalar(&[2, 2], 2.0, false);
        let target = Tensor::from_scalar(&[2, 2], 3.0, false);
        let grad = Tensor::from_scalar(&[2, 2], 1.0, false);
        let result = super::binary_cross_entropy_backward(&grad, &input, &target, None, 0);
        println!("BCE Result: {:?}", result);
    }
}
