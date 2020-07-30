use crate::variable::*;
use reduce::Reduce;
use std::rc::Rc;
pub trait Node {
    fn call(&mut self, inputs: Vec<Tensor>) -> Vec<Tensor> {
        self.apply(inputs)
    }
    fn apply(&mut self, input: Vec<Tensor>) -> Vec<Tensor>;
}

pub trait Function: Node {}
pub struct AccumulateGrad {
    tensor: Tensor,
}

impl AccumulateGrad {
    pub fn new(tensor: Tensor) -> Self {
        Self { tensor }
    }
}
impl Node for AccumulateGrad {
    fn apply(&mut self, grads: Vec<Tensor>) -> Vec<Tensor> {
        let new_grad = grads.into_iter().reduce(|a, b| a + b).unwrap();
        let grad = self.tensor.grad();
        if let Some(g) = grad {
            let t = unsafe { &*Rc::into_raw(g) };
            self.tensor.set_grad(t + new_grad);
        } else {
            self.tensor.set_grad(new_grad);
        }
        Vec::new()
    }
}
