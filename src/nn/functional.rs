use crate::tensor::{sigmoid, NewTensor};

pub struct Functional {
    fn_: Box<dyn Fn(&[&NewTensor]) -> NewTensor>,
}

impl std::fmt::Debug for Functional {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl Functional {
    pub fn new(func: Box<dyn Fn(&[&NewTensor]) -> NewTensor>) -> Self {
        Self { fn_: func }
    }

    pub fn sigmoid() -> Box<dyn Fn(&[&NewTensor]) -> NewTensor> {
        Box::new(|t: &[&NewTensor]| sigmoid(t[0]))
    }
}

impl super::Module for Functional {
    fn forward(&self, xs: &[&NewTensor]) -> NewTensor {
        (self.fn_)(xs)
    }

    fn parameters(&self) -> Vec<NewTensor> {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use super::super::Module;
    use super::Functional;
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
