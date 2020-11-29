use crate::aten::native;
use crate::c10::Scalar;
use crate::tensor::Tensor;

pub fn addmm(self_: &Tensor, mat1: &Tensor, mat2: &Tensor, beta: Scalar, alpha: Scalar) -> Tensor {
    native::addmm_cpu(self_, mat1, mat2, beta.into(), alpha.into())
}
