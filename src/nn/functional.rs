use crate::tensor::Tensor;

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
        Box::new(|t: &[&Tensor]| crate::ops::sigmoid(t[0]))
    }
}

impl super::Module for Functional {
    fn forward(&self, xs: &[&Tensor]) -> Tensor {
        (self.fn_)(xs)
    }
}

#[cfg(test)]
mod test {
    use super::Functional;
    use crate::tensor::Tensor;
    use super::super::Module;
    #[test]
    fn test_functional_sigmoid() {
        let sigmoid = Functional::sigmoid();
        let f = Functional::new(sigmoid);
        let x = Tensor::from_scalar(&[3,3], 2.0, false);
        let result = f.forward(&[&x]);
        println!("Result from Functional Layer with Sigmoid: {:?}", result);
    }
}
