#![allow(
    dead_code,
    non_camel_case_types,
    non_upper_case_globals,
    non_snake_case
)]
#![feature(min_const_generics, trace_macros)]

// trace_macros!(true);
extern crate openblas_src;

pub mod aten;
pub mod autograd;
pub mod c10;
mod core;
mod engine;
mod nn;
mod ops;
mod optim;
pub mod tensor;
mod util;
mod util_autograd;

use std::marker::PhantomData;
pub struct Closure<I, O, F, const N: usize>
where
    F: FnMut([I; N]) -> O,
{
    f: F,
    pub arity: usize,
    _in: PhantomData<I>,
    _out: PhantomData<O>,
}

impl<I, O, F, const N: usize> Closure<I, O, F, N>
where
    F: FnMut([I; N]) -> O,
{
    pub fn new(f: F) -> Self {
        Self {
            f,
            arity: N,
            _in: PhantomData,
            _out: PhantomData,
        }
    }
    pub fn call(&mut self, args: [I; N]) -> O {
        (self.f)(args)
    }
}

pub fn init_rovo() {
    c10::init();
}

#[cfg(test)]
mod test {
    use super::autograd::*;
    use super::c10::*;
    use super::tensor::*;
    #[test]
    fn test_backward_add() {
        crate::init_rovo();
        let t = ones(&[2, 2], TensorOptions::with_requires_grad());
        let x = ones(&[2, 2], TensorOptions::with_requires_grad());
        let res: Tensor = &t + &x;
        backward(&vec![res], &vec![], false);
        println!("{:?}", t.grad());
        println!("{:?}", x.grad());
    }
    #[test]
    fn test_backward_mul() {
        crate::init_rovo();
        let t = full(&[2, 2], 7.0, TensorOptions::with_requires_grad());
        let x = full(&[2, 2], 3.0, TensorOptions::with_requires_grad());
        let res: Tensor = &t * &x;
        dbg!(&res);
        backward(&vec![res], &vec![], false);
        println!("{:?}", t.grad());
        println!("{:?}", x.grad());
    }

    #[test]
    fn test_backward_mul_add() {
        crate::init_rovo();
        let t = full(&[2, 2], 7.0, TensorOptions::with_requires_grad());
        let x = full(&[2, 2], 3.0, TensorOptions::with_requires_grad());
        let add: Tensor = &t + &x;
        let mul = &x * &add;
        backward(&vec![mul], &vec![], false);
        println!("{:?}", t.grad());
        println!("{:?}", x.grad());
    }

    // #[test]
    // fn test_backward_sub_div() {
    //     let t = Tensor::from_scalar(&[2, 2], 7.0, true);
    //     let x = Tensor::from_scalar(&[2, 2], 3.0, true);
    //     let div = &t / &x;
    //     let sub = &x - &div;
    //     backward(&vec![sub], &vec![], false);
    //     // t.grad().unwrap();
    //     println!("{:?}", t.grad().unwrap().as_ref());
    //     println!("{:?}", x.grad().unwrap().as_ref());
    // }

    // #[test]
    // fn test_matmul() {
    //     let x = Tensor::from_scalar(&[1, 2], 2.0, true);
    //     let w = Tensor::from_scalar(&[1, 2], 3.0, true);
    //     let b = Tensor::from_scalar(&[1], 3.0, true);
    //     let result = x.matmul(&w.t(), true) + b;
    //     backward(&vec![result], &vec![], false);
    //     // t.grad().unwrap();
    //     println!("{:?}", x.grad().unwrap().as_ref());
    //     println!("{:?}", w.grad().unwrap().as_ref());
    // }

    #[test]
    fn test_sigmoid_backward() {
        let input = ones(&[2, 3], TensorOptions::with_requires_grad());
        let result = sigmoid(&input);
        println!("Result: {:?}", result);
        backward(&vec![result], &vec![], false);
        println!("Input Grad: {:?}", input.grad());
    }
}
