use crate::tensor::*;

pub trait Module: std::fmt::Debug + Send {
    fn forward(&self, xs: &Tensor) -> Tensor;
}

pub trait ModuleT: std::fmt::Debug + Send {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor;
}

impl<T> ModuleT for T
where
    T: Module,
{
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(&xs)
    }
}

// struct Model{
//     pub parameters: OrderedDict<String, Tensor>,
//     buffers: OrderedDict<String, Tensor>,
//     children: OrderedDict<String, Rc<RefCell<dyn Module>>>,
//     name: Option<String>,
//     is_training: bool
// }

// impl Default for Model{
//     fn default() -> Self {
//         Self{
//             parameters: OrderedDict::new_with_key_description(String::from("Parameter")),
//             buffers: OrderedDict::new_with_key_description(String::from("Buffer")),
//             children: OrderedDict::new_with_key_description(String::from("children")),
//             name: None,
//             is_training: true
//         }
//     }
// }

// impl Model{
//     pub fn new_with_name(name: String)->Self{
//         Self{name: Some(name), ..Default::default()}
//     }
// }

// impl Module for Model{
//     fn name(&self) ->String {
//         match &self.name {
//             Some(n)=>n.clone(),
//             // infername of the impl similar to C++ typeid().name().
//             None=> String::from("Unnamed_Model"),
//         }
//     }
// }

impl Tensor {
    pub fn apply<M: Module>(&self, m: &M) -> Tensor {
        m.forward(&self)
    }

    pub fn apply_t<M: ModuleT>(&self, m: &M, train: bool) -> Tensor {
        m.forward_t(&self, train)
    }

    pub fn apply_opt<M: Module>(&self, m: &Option<M>) -> Tensor {
        match m {
            Some(m) => m.forward(&self),
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


pub fn register_parameter(tensor: &Tensor, requires_grad: bool){
    tensor.set_requires_grad(requires_grad);
}
#[cfg(test)]
mod tests {
    // use super::Model;

    #[test]
    fn it_works() {}
}
