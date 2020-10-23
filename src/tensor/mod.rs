mod linear_algebra;
pub mod loss;
mod tensor;
mod tensor_impl;
mod tensor_iterator;
mod tensor_ops;
mod tensor_options;
mod tensor_util;

// pub use self::loss;
pub use tensor::*;
pub use tensor_impl::*;
pub use tensor_iterator::*;
pub use tensor_ops::*;
pub use tensor_options::*;
pub use tensor_util::*;

pub type VariableList = Vec<Tensor>;
pub type EdgeList = Vec<Edge>;

//Improve debug impl
impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", unsafe { &*self._impl.as_ptr() }.data)
    }
}
