use super::{Tensor, TensorImpl};
use crate::ndarry_ext;

pub fn matmul(tensor1: &Tensor, tensor2: &Tensor, consume: bool) -> Tensor {
    let _impl1 = tensor1.get_tensor_impl();
    let _impl2 = tensor2.get_tensor_impl();
    let dim1 = tensor1.dim();
    let dim2 = tensor2.dim();

    let out = match (dim1, dim2) {
        (1, 1) => tensor1.dot(tensor2),
        (2, 2) => tensor1.mm(tensor2, consume),
        (_, _) => todo!(),
    };

    out
}

pub fn dot(_tensor1: &Tensor, _tensor2: &Tensor) -> Tensor {
    todo!()
}

pub fn mm(_tensor1: &Tensor, _tensor2: &Tensor) -> Tensor {
    let mat1 = _tensor1._impl.borrow();
    let mat2 = _tensor2._impl.borrow();

    assert_eq!(mat1.data.ndim(), 2);
    assert_eq!(mat2.data.ndim(), 2);

    let result = ndarry_ext::dot(&mat1.data, &mat2.data);
    Tensor::from_impl(TensorImpl::new_from_array(result, false))
}

pub fn mm_consume(_tensor1: &Tensor, _tensor2: Tensor) -> Tensor {
    let mat1 = _tensor1._impl.borrow();
    let mat2 = _tensor2._impl.borrow();

    assert_eq!(mat1.data.ndim(), 2);
    assert_eq!(mat2.data.ndim(), 2);

    let result = ndarry_ext::dot(&mat1.data, &mat2.data);
    Tensor::from_impl(TensorImpl::new_from_array(result, false))
}