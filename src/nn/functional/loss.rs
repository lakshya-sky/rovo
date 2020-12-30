use crate::tensor::Tensor;
use crate::{aten::native, tensor::loss::Reduction};

pub struct NLLLossOptions {
    /// A manual rescaling weight given to each
    /// class. If given, it has to be a Tensor of size `C`. Otherwise, it is
    /// treated as if having all ones.
    weight: Option<Tensor>,
    reduction: Reduction,
    ignore_index: i64,
}

impl Default for NLLLossOptions {
    fn default() -> Self {
        Self {
            reduction: Reduction::Mean,
            weight: None,
            ignore_index: -100,
        }
    }
}

impl NLLLossOptions {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn set_weight(&mut self, tensor: Tensor) -> &mut Self {
        if tensor.defined() {
            self.weight = Some(tensor);
        } else {
            self.weight = None;
        }
        self
    }
    pub fn set_reduction(&mut self, reduction: Reduction) -> &mut Self {
        self.reduction = reduction;
        self
    }

    pub fn set_ignore_index(&mut self, index: i64) -> &mut Self {
        self.ignore_index = index;
        self
    }

    pub fn weight(&self) -> Option<&Tensor> {
        self.weight.as_ref()
    }
    pub fn reduction(&self) -> Reduction {
        self.reduction
    }
    pub fn ignore_index(&self) -> i64 {
        self.ignore_index
    }
}
impl AsRef<NLLLossOptions> for NLLLossOptions {
    fn as_ref(&self) -> &NLLLossOptions {
        self
    }
}

pub type NLLLossFuncOptions = NLLLossOptions;

#[inline(always)]
pub fn nll_loss<O: AsRef<NLLLossFuncOptions>>(
    input: &Tensor,
    target: &Tensor,
    options: O,
) -> Tensor {
    let options = options.as_ref();
    native::nll_loss(
        input,
        target,
        options.weight(),
        options.reduction(),
        options.ignore_index(),
    )
}
