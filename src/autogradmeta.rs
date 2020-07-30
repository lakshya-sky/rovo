use crate::variable::*;
use crate::ops::*;
use std::rc::Rc;
use std::cell::RefCell;
pub struct AutogradMeta {
    pub grad_fn_: Option<Rc<RefCell<dyn Node>>>,
    pub grad_accumulator_: Option<Rc<RefCell<dyn Node>>>,
    pub grad_: Option<Rc<Tensor>>,
}

impl AutogradMeta{
    pub fn grad(&self) -> Option<Rc<Tensor>> {
        self.grad_.clone()
    }
    pub fn set_grad(&mut self, grad: Tensor){
        self.grad_ = Some(Rc::new(grad))
    }
}

pub struct AutogradMetaFactory{

}

impl AutogradMetaFactory{
    pub fn make() -> AutogradMeta{
        AutogradMeta{
            grad_: None,
            grad_fn_: None,
            grad_accumulator_: None,
        }
    }
}