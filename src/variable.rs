use std::cell::RefCell;
use std::rc::Rc;
use crate::ops;
pub struct Variable {
    pub data: f64,
    pub grad: Option<f64>,
    pub grad_fn: Option<Rc<RefCell<dyn ops::GradFunction>>>,
}

impl Variable {
    pub fn backward(&mut self, grad_arg: Option<&Variable>) {
        let mut grad = &Variable {
            data: 1.0,
            grad: None,
            grad_fn: None,
        };
        if let Some(g) = grad_arg {
            grad = g;
        }
        self.grad = Some(grad.data);
        if let Some(grad_fn) = self.grad_fn.as_ref() {
            grad_fn.borrow().apply(grad);
        }
    }
}