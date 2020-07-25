use crate::ndarry_ext::*;
use crate::variable::*;
use std::cell::RefCell;
// use std::fmt;
// use std::ops;
use std::rc::Rc;
// use crate::binary_ops;

pub trait Function<T> {
    fn apply(&self, a: &Variable<T>, b: &Variable<T>) -> Variable<T> {
        Self::forward(a, b)
    }
    fn forward(a: &Variable<T>, b: &Variable<T>) -> Variable<T>;
}
pub trait GradFunction<T> {
    fn apply(&self, grad: &NdArray<T>);
}

pub struct Add;
pub struct Mul;

impl<T: Float> Function<T> for Add {
    fn forward(a: &Variable<T>, b: &Variable<T>) -> Variable<T> {
        let a_borrowed = a.data.as_ref().as_ptr();
        let b_borrowed = b.data.as_ref().as_ptr();

        let data = unsafe { a_borrowed.as_ref().unwrap() + b_borrowed.as_ref().unwrap() };
        // let grad_fn = AddBackward {
        //     x: Rc::clone(a),
        //     y: Rc::clone(b),
        // };
        Variable {
            data: Rc::new(RefCell::new(data)),
            grad: None,
            grad_fn: None,
        }
    }
}

// impl<T> Function<T> for Mul {
//     fn forward(a: &Rc<RefCell<Variable<T>>>, b: &Rc<RefCell<Variable<T>>>) -> Variable<T> {
//         let a_borrowed = unsafe{(*a.as_ptr()).data.get_mut()};
//         let b_borrowed = unsafe{(*b.as_ptr()).data.get_mut()};

//         let data = a_borrowed * b_borrowed;
//         let grad_fn = MulBackward {
//             x: Rc::clone(a),
//             y: Rc::clone(b),
//         };
//         Variable {
//             data,
//             grad: None,
//             grad_fn: Some(Rc::new(RefCell::new(grad_fn))),
//         }
//     }
// }

// pub struct AddBackward<T> {
//     x: Rc<RefCell<Variable<T>>>,
//     y: Rc<RefCell<Variable<T>>>,
// }

// impl<T> GradFunction<T> for AddBackward<T> {
//     fn apply(&self, grad: &Variable<T>) {
//         self.x.borrow_mut().backward(Some(&Variable {
//             data: grad.data,
//             grad: None,
//             grad_fn: None,
//         }));
//         self.y.borrow_mut().backward(Some(&Variable {
//             data: grad.data,
//             grad: None,
//             grad_fn: None,
//         }));
//     }
// }
// pub struct MulBackward<T> {
//     x: Rc<RefCell<Variable<T>>>,
//     y: Rc<RefCell<Variable<T>>>,
// }

// impl<T> GradFunction<T> for MulBackward<T> {
//     fn apply(&self, grad: &Variable<T>) {
//         self.x.borrow_mut().backward(Some(&Variable {
//             data: grad.data * self.y.borrow().data,
//             grad: None,
//             grad_fn: None,
//         }));
//         self.y.borrow_mut().backward(Some(&Variable {
//             data: grad.data * self.x.borrow().data,
//             grad: None,
//             grad_fn: None,
//         }));
//     }
// }

// impl<T> fmt::Debug for Variable<T> {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         f.debug_struct("Variable")
//             .field("data", &self.data)
//             .finish()
//     }
// }
// impl<T> fmt::Display for Variable<T> {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         f.debug_struct("Variable")
//             .field("data", &self.data)
//             .finish()
//     }
// }

// impl<T> ops::Add<Self> for Variable<T> {
//     type Output = Self;
//     fn add(self, rhs:Self) -> Self::Output {
//         let rc_lhs = Rc::new(RefCell::new(Self{
//             data: self.data,
//             grad: self.grad,
//             grad_fn: self.grad_fn
//         }));
//         let rc_rhs = Rc::new(RefCell::new(Self{
//             data: rhs.data,
//             grad: rhs.grad,
//             grad_fn: rhs.grad_fn
//         }));

//         let op = Add{};
//         let out = op.apply(&rc_lhs, &rc_rhs);
//         out
//     }
// }

// impl<T> ops::Mul<Self> for Variable<T> {
//     type Output = Self;
//     fn mul(self, rhs:Self) -> Self::Output {
//         let rc_lhs = Rc::new(RefCell::new(Self{
//             data: self.data,
//             grad: self.grad,
//             grad_fn: self.grad_fn
//         }));
//         let rc_rhs = Rc::new(RefCell::new(Self{
//             data: rhs.data,
//             grad: rhs.grad,
//             grad_fn: rhs.grad_fn
//         }));

//         let op = Mul{};
//         let out = op.apply(&rc_lhs, &rc_rhs);
//         out
//     }
// }

#[cfg(test)]
mod tests {
    use crate::ops::*;
    use crate::variable::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn test_add_op() {
        let a = Variable::<f64>::ones(&[2, 2]);
        let b = Variable::<f64>::ones(&[2, 2]);
        let op = Add {};
        let out = op.apply(&a, &b);
        assert_eq!(&[2.0; 4], out.data.borrow().as_slice().unwrap());
        // println!("{:?}", out.data.borrow().as_slice().unwrap())
    }
//     #[test]
//     fn test_add_backward() {
//         let a = Rc::new(RefCell::new(Variable {
//             data: 2.0,
//             grad: None,
//             grad_fn: None,
//         }));
//         let b = Rc::new(RefCell::new(Variable {
//             data: 3.0,
//             grad: None,
//             grad_fn: None,
//         }));
//         let op = Add {};
//         let mut out = op.apply(&a, &b);
//         out.backward(None);
//         assert_eq!(1.0, a.borrow().grad.unwrap());
//         assert_eq!(1.0, b.borrow().grad.unwrap());
//     }
//     #[test]
//     fn test_add_overloading(){
//         let a = Variable{data: 2.0, grad: None, grad_fn: None};
//         let b = Variable{data: 3.0, grad: None, grad_fn: None};
//         let c: Variable = a+b;
//         assert_eq!(5.0, c.data);
//     }
//     #[test]
//     fn test_mul_op() {
//         let a = Rc::new(RefCell::new(Variable {
//             data: 2.0,
//             grad: None,
//             grad_fn: None,
//         }));
//         let b = Rc::new(RefCell::new(Variable {
//             data: 3.0,
//             grad: None,
//             grad_fn: None,
//         }));
//         let op = Mul {};
//         let out = op.apply(&a, &b);
//         assert_eq!(6.0, out.data);
//     }
//     #[test]
//     fn test_mul_backward() {
//         let a = Rc::new(RefCell::new(Variable {
//             data: 2.0,
//             grad: None,
//             grad_fn: None,
//         }));
//         let b = Rc::new(RefCell::new(Variable {
//             data: 3.0,
//             grad: None,
//             grad_fn: None,
//         }));
//         let op = Mul {};
//         let mut out = op.apply(&a, &b);
//         out.backward(None);
//         assert_eq!(3.0, a.borrow().grad.unwrap());
//         assert_eq!(2.0, b.borrow().grad.unwrap());
//     }
//     #[test]
//     fn test_mul_overloading(){
//         let a = Variable{data: 2.0, grad: None, grad_fn: None};
//         let b = Variable{data: 3.0, grad: None, grad_fn: None};
//         let c: Variable = a*b;
//         assert_eq!(6.0, c.data);
//     }
}
