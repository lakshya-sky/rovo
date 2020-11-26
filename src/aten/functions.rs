use crate::aten::native;
use crate::c10::Scalar;
use crate::tensor::Tensor;

pub fn addmm<S: Into<Scalar>>(
    self_: &Tensor,
    mat1: &Tensor,
    mat2: &Tensor,
    beta: S,
    alpha: S,
) -> Tensor {
    native::addmm_cpu(self_, mat1, mat2, beta.into(), alpha.into())
}
