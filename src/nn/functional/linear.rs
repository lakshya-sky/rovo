use crate::aten;
use crate::tensor::Tensor;

pub fn linear(input: &Tensor, weight: &Tensor, bias: &Tensor) -> Tensor {
    if input.dim() == 2 && bias.defined() {
        aten::addmm(bias, input, &weight.t(), 1.0, 1.0)
    } else {
        todo!()
    }
}
