mod linear_algebra;
mod tensor;
mod tensor_impl;
mod tensor_ops;
mod tensor_util;

pub use self::tensor::*;
pub use self::tensor_impl::*;
pub use self::tensor_ops::*;
pub use self::tensor_util::*;
pub type VariableList = Vec<Tensor>;
pub type EdgeList = Vec<Edge>;

//Improve debug impl
impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", unsafe { &*self._impl.as_ptr() }.data)
    }
}
