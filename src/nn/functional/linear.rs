use crate::tensor::{addmm, Tensor};

pub fn linear(input: &Tensor, weight: &Tensor, bias: &Tensor) -> Tensor {
    if input.dim() == 2 && bias.defined() {
        addmm(bias, input, &weight.t(), 1.0, 1.0)
    } else {
        todo!()
    }
}
