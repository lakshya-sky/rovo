mod tensor_impl;
mod tensor;

pub use self::tensor::*;
pub use self::tensor_impl::*;
pub type VariableList = Vec<Tensor>;
pub type EdgeList = Vec<Edge>;
