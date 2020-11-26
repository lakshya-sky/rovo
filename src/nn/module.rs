use crate::tensor::*;

pub trait Module: std::fmt::Debug {
    fn forward(&self, xs: &[&Tensor]) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;
}

pub trait ModuleT: std::fmt::Debug {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor;
}

impl<T> ModuleT for T
where
    T: Module,
{
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(&[xs])
    }
}

impl Tensor {
    pub fn apply<M: Module>(&self, m: &M) -> Tensor {
        m.forward(&[self])
    }

    pub fn apply_t<M: ModuleT>(&self, m: &M, train: bool) -> Tensor {
        m.forward_t(&self, train)
    }

    pub fn apply_opt<M: Module>(&self, m: &Option<M>) -> Tensor {
        match m {
            Some(m) => m.forward(&[self]),
            None => self.clone(),
        }
    }

    pub fn apply_opt_t<M: ModuleT>(&self, m: &Option<M>, train: bool) -> Tensor {
        match m {
            Some(m) => m.forward_t(&self, train),
            None => self.clone(),
        }
    }
}

pub fn register_parameter(Tensor: &Tensor, requires_grad: bool) {
    Tensor.set_requires_grad(requires_grad);
}
#[cfg(test)]
mod tests {
    // use super::Model;

    #[test]
    fn it_works() {}
}
