use crate::ndarry_ext::*;
use crate::ops;
use std::cell::RefCell;
use std::rc::Rc;
pub struct Variable<T> {
    pub data: Rc<RefCell<NdArray<T>>>,
    pub grad: Option<NdArray<T>>,
    pub grad_fn: Option<Rc<RefCell<dyn ops::GradFunction<T>>>>,
}
impl<T: Float> Variable<T> {
    pub fn ones(shape: &[usize]) -> Variable<T> {
        Variable {
            data: Rc::new(RefCell::new(NdArray::<T>::ones(shape))),
            grad: None,
            grad_fn: None,
        }
    }
    pub fn backward(&mut self, grad_arg: Option<&NdArray<T>>) {
        if let Some(g) = grad_arg {
            self.grad = Some(g.clone());
        } else {
            self.grad = Some(NdArray::<T>::ones(self.data.borrow().shape()));
        }
        if let Some(grad_fn) = self.grad_fn.as_ref() {
            grad_fn.borrow().apply(self.grad.as_ref().unwrap());
        }
    }
}
