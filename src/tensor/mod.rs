mod linear_algebra;
pub mod loss;
mod tensor;
mod tensor_impl;
mod tensor_iterator;
mod tensor_oprators;
mod tensor_ops;
mod tensor_options;
mod tensor_shape;
mod tensor_util;
// pub use self::loss;
pub use tensor::*;
pub use tensor_impl::*;
pub use tensor_iterator::*;
pub use tensor_oprators::*;
pub use tensor_ops::*;
pub use tensor_options::*;
pub use tensor_shape::*;
pub use tensor_util::*;
pub type VariableList = Vec<NewTensor>;
pub type EdgeList = Vec<Edge>;

//Improve debug impl
impl std::fmt::Debug for NewTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.defined() {
            write!(f, "{:?}", self.get_unsafe_tensor_impl())
        } else {
            write!(f, "Undefined Tensor")
        }
    }
}
#[cfg(test)]
mod test {
    use crate::autograd;

    #[test]
    fn empty_tensor_and_fill_ones() {
        crate::init_rovo();
        let t = autograd::empty(&[2, 2], None, None);
        t.fill_(1.0);
        println!("{:?}", t);
    }

    #[test]
    fn test_fill_and_add() {
        crate::init_rovo();
        let t = autograd::empty(&[2, 2], None, None);
        let r = autograd::empty(&[2, 2], None, None);
        t.fill_(1.0);
        r.fill_(1.23);
        t.add_(&r, 0.0);
        println!("{:?}", t);
    }
}
