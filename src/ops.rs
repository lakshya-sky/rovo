use crate::ndarry_ext::*;
use crate::variable::*;
use std::cell::RefCell;
// use std::fmt;
// use std::ops;
use std::rc::Rc;
// use crate::binary_ops;

pub trait Function<T> {
    fn apply(&self, a: &Rc<RefCell<VariableImpl<T>>>, b: &Rc<RefCell<VariableImpl<T>>>) -> VariableImpl<T> {
        Self::forward(a, b)
    }
    fn forward(a: &Rc<RefCell<VariableImpl<T>>>, b: &Rc<RefCell<VariableImpl<T>>>) -> VariableImpl<T>;
}
pub trait GradFunction<T> {
    fn apply(&mut self, grad: &NdArray<T>);
}

pub struct Add;
pub struct Mul;

impl<T: Float> Function<T> for Add {
    fn forward(a: &Rc<RefCell<VariableImpl<T>>>, b: &Rc<RefCell<VariableImpl<T>>>) -> VariableImpl<T> {
        let a_borrowed = &a.borrow().data;
        let b_borrowed = &b.borrow().data;

        let data = a_borrowed + b_borrowed;
        let grad_fn = AddBackward::new(a.clone(), b.clone());
        VariableImpl {
            data: data,
            grad: None,
            grad_fn: Some(Rc::new(RefCell::new(grad_fn))),
        }
    }
}

impl<T: Float> Function<T> for Mul {
    fn forward(a: &Rc<RefCell<VariableImpl<T>>>, b: &Rc<RefCell<VariableImpl<T>>>) -> VariableImpl<T> {
        let a_borrowed = &a.borrow().data;
        let b_borrowed = &b.borrow().data;

        let data = a_borrowed * b_borrowed;
        let grad_fn = MulBackward {
            x: a.clone(),
            y: b.clone(),
        };
        VariableImpl {
            data,
            grad: None,
            grad_fn: Some(Rc::new(RefCell::new(grad_fn))),
        }
    }
}

pub struct AddBackward<T> {
    x: Rc<RefCell<VariableImpl<T>>>,
    y: Rc<RefCell<VariableImpl<T>>>,
}

impl<T> AddBackward<T> {
    pub fn new(x: Rc<RefCell<VariableImpl<T>>>, y: Rc<RefCell<VariableImpl<T>>>) -> Self {
        AddBackward { x, y }
    }
}

impl<T> GradFunction<T> for AddBackward<T>
where
    T: Float,
{
    fn apply(&mut self, grad: &NdArray<T>) {
        self.x.borrow_mut().backward(Some(grad));
        self.y.borrow_mut().backward(Some(grad));
    }
}
pub struct MulBackward<T> {
    x: Rc<RefCell<VariableImpl<T>>>,
    y: Rc<RefCell<VariableImpl<T>>>,
}

impl<T> GradFunction<T> for MulBackward<T>
where
    T: Float,
{
    fn apply(&mut self, grad: &NdArray<T>) {
        self.x
            .borrow_mut()
            .backward(Some(&(grad * &(self.y.borrow().data))));
        self.y
            .borrow_mut()
            .backward(Some(&(grad * &(self.x.borrow().data))));
    }
}

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
    use crate::variable::VariableImpl;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn test_add_op() {
        let a = Rc::new(RefCell::new(VariableImpl::<f64>::ones(&[2, 2])));
        let b = Rc::new(RefCell::new(VariableImpl::<f64>::ones(&[2, 2])));
        let op = Add {};
        let out = op.apply(&a, &b);
        assert_eq!(&[2.0; 4], out.data.as_slice().unwrap());
        // println!("{:?}", out.data.borrow().as_slice().unwrap())
    }
    #[test]
    fn test_add_backward() {
        let mut a = Rc::new(RefCell::new(VariableImpl::from_constant(&[2, 2], 3.0)));
        let mut b = Rc::new(RefCell::new(VariableImpl::from_constant(&[2, 2], 2.0)));
        let op = Add {};
        let mut out = op.apply(&mut a, &mut b);
        out.backward(None);
        assert_eq!(
            &[1.0; 4],
            a.borrow().grad.as_ref().unwrap().as_slice().unwrap()
        );
        assert_eq!(
            &[1.0; 4],
            b.borrow().grad.as_ref().unwrap().as_slice().unwrap()
        );
    }
    #[test]
    fn test_mul_op() {
        let a = Rc::new(RefCell::new(VariableImpl::<f64>::ones(&[2, 2])));
        let b = Rc::new(RefCell::new(VariableImpl::<f64>::ones(&[2, 2])));
        let op = Mul {};
        let out = op.apply(&a, &b);
        assert_eq!(&[1.0; 4], out.data.as_slice().unwrap());
        // println!("{:?}", out.data.borrow().as_slice().unwrap())
    }
    #[test]
    fn test_mul_backward() {
        let mut a = Rc::new(RefCell::new(VariableImpl::from_constant(&[2, 2], 3.0)));
        let mut b = Rc::new(RefCell::new(VariableImpl::from_constant(&[2, 2], 2.0)));
        let op = Mul {};
        let mut out = op.apply(&mut a, &mut b);
        out.backward(None);
        assert_eq!(
            &[2.0; 4],
            a.borrow().grad.as_ref().unwrap().as_slice().unwrap()
        );
        assert_eq!(
            &[3.0; 4],
            b.borrow().grad.as_ref().unwrap().as_slice().unwrap()
        );
    }
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
