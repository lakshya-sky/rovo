use crate::ndarry_ext::*;
use crate::ops::*;
use std::cell::RefCell;
use std::rc::Rc;

pub struct VariableImpl<T> {
    pub data: NdArray<T>,
    pub grad: Option<NdArray<T>>,
    pub grad_fn: Option<Rc<RefCell<dyn GradFunction<T>>>>,
}

impl<T: Float> VariableImpl<T> {
    pub fn ones(shape: &[usize]) -> VariableImpl<T> {
        VariableImpl {
            data: NdArray::<T>::ones(shape),
            grad: None,
            grad_fn: None,
        }
    }

    pub fn from_constant(shape: &[usize], constant: T) -> VariableImpl<T> {
        VariableImpl {
            data: NdArray::<T>::from_elem(shape, constant),
            grad: None,
            grad_fn: None,
        }
    }

    pub fn backward(&mut self, grad_arg: Option<&NdArray<T>>) {
        if let Some(g) = grad_arg {
            self.grad = Some(g.clone());
        } else {
            self.grad = Some(NdArray::<T>::ones(self.data.shape()));
        }
        if let Some(grad_fn) = self.grad_fn.clone() {
            grad_fn.borrow_mut().apply(self.grad.as_ref().unwrap());
        }
    }
}
