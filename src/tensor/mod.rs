pub mod loss;
mod tensor;
mod tensor_impl;
mod tensor_iterator;
mod tensor_oprators;
mod tensor_ops;
mod tensor_shape;
mod tensor_util;
// pub use self::loss;
pub use tensor::*;
pub use tensor_impl::*;
pub use tensor_iterator::*;
pub use tensor_oprators::*;
pub use tensor_ops::*;
pub use tensor_shape::*;
pub use tensor_util::*;
pub type VariableList = Vec<Tensor>;
pub type EdgeList = Vec<Edge>;

//Improve debug impl
impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.defined() {
            write!(
                f,
                "Tensor: {:?}\tsize: {:?}",
                self.get_unsafe_tensor_impl(),
                self.sizes()
            )
        } else {
            write!(f, "Undefined Tensor")
        }
    }
}
