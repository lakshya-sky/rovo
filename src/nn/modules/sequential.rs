use std::collections::VecDeque;

use crate::{nn::Module, tensor::Tensor};
#[derive(Debug)]
pub struct Sequential {
    modules: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new() -> Self {
        Self {
            modules: Vec::default(),
        }
    }

    pub fn add<M>(&mut self, module: M) -> &mut Self
    where
        M: Module + 'static,
    {
        self.modules.push(Box::new(module));
        self
    }
}

impl Module for Sequential {
    fn forward(&self, input: &[&Tensor]) -> Tensor {
        let mut output = self.modules.first().unwrap().forward(input);
        for m in self.modules[1..].iter() {
            output = m.forward(&[&output]);
        }
        output
    }

    fn parameters(&self) -> Option<Vec<Tensor>> {
        let mut result = vec![];
        for module in self.modules.iter() {
            if let Some(mut param_vec) = module.parameters() {
                param_vec.drain(..).for_each(|p| result.push(p))
            }
        }
        Some(result)
    }
}
