pub mod loss;
mod tensor_;
mod tensor_impl;
mod tensor_iterator;
mod tensor_oprators;
mod tensor_ops;
mod tensor_shape;
mod tensor_util;
// pub use self::loss;
pub use tensor_::*;
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
            let tensor = self.to_dtype(crate::c10::ScalarType::Double);
            write!(
                f,
                "Tensor: {:?}\tsize: {:?}",
                tensor.get_unsafe_tensor_impl(),
                tensor.sizes()
            )
        } else {
            write!(f, "Undefined Tensor")
        }
    }
}
