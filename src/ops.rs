use crate::variable::*;
use std::cell::RefCell;
use std::rc::Rc;

pub trait Function {
    fn apply(&self, a: &Rc<RefCell<Variable>>, b: &Rc<RefCell<Variable>>) -> Variable{
        Self::forward(a,b)
    }
    fn forward(a: &Rc<RefCell<Variable>>, b: &Rc<RefCell<Variable>>) -> Variable;
}
pub trait GradFunction {
    fn apply(&self, grad: &Variable);
}


pub struct Add;
pub struct Mul;

impl Function for Add {
    fn forward(a: &Rc<RefCell<Variable>>, b: &Rc<RefCell<Variable>>) -> Variable {
        let data = a.borrow().data + b.borrow().data;
        let grad_fn = AddBackward {
            x: Rc::clone(a),
            y: Rc::clone(b),
        };
        Variable {
            data,
            grad: None,
            grad_fn: Some(Rc::new(RefCell::new(grad_fn))),
        }
    }
}

impl Function for Mul {
    fn forward(a: &Rc<RefCell<Variable>>, b: &Rc<RefCell<Variable>>) -> Variable {
        let data = a.borrow().data * b.borrow().data;
        let grad_fn = MulBackward {
            x: Rc::clone(a),
            y: Rc::clone(b),
        };
        Variable {
            data,
            grad: None,
            grad_fn: Some(Rc::new(RefCell::new(grad_fn))),
        }
    }
}

pub struct AddBackward {
    x: Rc<RefCell<Variable>>,
    y: Rc<RefCell<Variable>>,
}

impl GradFunction for AddBackward {
    fn apply(&self, grad: &Variable) {
        self.x.borrow_mut().backward(Some(&Variable {
            data: grad.data,
            grad: None,
            grad_fn: None,
        }));
        self.y.borrow_mut().backward(Some(&Variable {
            data: grad.data,
            grad: None,
            grad_fn: None,
        }));
    }
}
pub struct MulBackward {
    x: Rc<RefCell<Variable>>,
    y: Rc<RefCell<Variable>>,
}

impl GradFunction for MulBackward {
    fn apply(&self, grad: &Variable) {
        self.x.borrow_mut().backward(Some(&Variable {
            data: grad.data * self.y.borrow().data,
            grad: None,
            grad_fn: None,
        }));
        self.y.borrow_mut().backward(Some(&Variable {
            data: grad.data * self.x.borrow().data,
            grad: None,
            grad_fn: None,
        }));
    }
}

#[cfg(test)]
mod tests {
    use crate::variable::*;
    use crate::ops::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn test_add_op() {
        let a = Rc::new(RefCell::new(Variable {
            data: 2.0,
            grad: None,
            grad_fn: None,
        }));
        let b = Rc::new(RefCell::new(Variable {
            data: 3.0,
            grad: None,
            grad_fn: None,
        }));
        let op = Add {};
        let out = op.apply(&a, &b);
        assert_eq!(5.0, out.data);
    }
    #[test]
    fn test_add_backward() {
        let a = Rc::new(RefCell::new(Variable {
            data: 2.0,
            grad: None,
            grad_fn: None,
        }));
        let b = Rc::new(RefCell::new(Variable {
            data: 3.0,
            grad: None,
            grad_fn: None,
        }));
        let op = Add {};
        let mut out = op.apply(&a, &b);
        out.backward(None);
        assert_eq!(1.0, a.borrow().grad.unwrap());
        assert_eq!(1.0, b.borrow().grad.unwrap());
    }
    #[test]
    fn test_mul_op() {
        let a = Rc::new(RefCell::new(Variable {
            data: 2.0,
            grad: None,
            grad_fn: None,
        }));
        let b = Rc::new(RefCell::new(Variable {
            data: 3.0,
            grad: None,
            grad_fn: None,
        }));
        let op = Mul {};
        let out = op.apply(&a, &b);
        assert_eq!(6.0, out.data);
    }
    #[test]
    fn test_mul_backward() {
        let a = Rc::new(RefCell::new(Variable {
            data: 2.0,
            grad: None,
            grad_fn: None,
        }));
        let b = Rc::new(RefCell::new(Variable {
            data: 3.0,
            grad: None,
            grad_fn: None,
        }));
        let op = Mul {};
        let mut out = op.apply(&a, &b);
        out.backward(None);
        assert_eq!(3.0, a.borrow().grad.unwrap());
        assert_eq!(2.0, b.borrow().grad.unwrap());
    }
}
