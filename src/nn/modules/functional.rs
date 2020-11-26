use crate::nn::module::Module;
use crate::tensor::{sigmoid, Tensor};

pub struct Functional {
    fn_: Box<dyn Fn(&[&Tensor]) -> Tensor>,
}

impl std::fmt::Debug for Functional {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl Functional {
    pub fn new(func: Box<dyn Fn(&[&Tensor]) -> Tensor>) -> Self {
        Self { fn_: func }
    }

    pub fn sigmoid() -> Box<dyn Fn(&[&Tensor]) -> Tensor> {
        Box::new(|t: &[&Tensor]| sigmoid(t[0]))
    }
}

impl Module for Functional {
    fn forward(&self, xs: &[&Tensor]) -> Tensor {
        (self.fn_)(xs)
    }

    fn parameters(&self) -> Vec<Tensor> {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use super::Functional;
    use super::Module;
    use crate::autograd::full;
    #[test]
    fn test_functional_sigmoid() {
        let sigmoid = Functional::sigmoid();
        let f = Functional::new(sigmoid);
        let x = full(&[3, 3], 2.0, None);
        let result = f.forward(&[&x]);
        println!("Result from Functional Layer with Sigmoid: {:?}", result);
    }
}
