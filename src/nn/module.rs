use crate::tensor::*;

pub trait Module: std::fmt::Debug {
    fn forward(&self, xs: &[&NewTensor]) -> NewTensor;
    fn parameters(&self) -> Vec<NewTensor>;
}

pub trait ModuleT: std::fmt::Debug {
    fn forward_t(&self, xs: &NewTensor, train: bool) -> NewTensor;
}

impl<T> ModuleT for T
where
    T: Module,
{
    fn forward_t(&self, xs: &NewTensor, _train: bool) -> NewTensor {
        self.forward(&[xs])
    }
}

impl NewTensor {
    pub fn apply<M: Module>(&self, m: &M) -> NewTensor {
        m.forward(&[self])
    }

    pub fn apply_t<M: ModuleT>(&self, m: &M, train: bool) -> NewTensor {
        m.forward_t(&self, train)
    }

    pub fn apply_opt<M: Module>(&self, m: &Option<M>) -> NewTensor {
        match m {
            Some(m) => m.forward(&[self]),
            None => self.clone(),
        }
    }

    pub fn apply_opt_t<M: ModuleT>(&self, m: &Option<M>, train: bool) -> NewTensor {
        match m {
            Some(m) => m.forward_t(&self, train),
            None => self.clone(),
        }
    }
}

pub fn register_parameter(newtensor: &NewTensor, requires_grad: bool) {
    newtensor.set_requires_grad(requires_grad);
}
#[cfg(test)]
mod tests {
    // use super::Model;

    #[test]
    fn it_works() {}
}
