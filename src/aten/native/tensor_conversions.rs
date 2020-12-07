use crate::{
    autograd::empty,
    c10::TensorOptions,
    c10::{MemoryFormat, ScalarType},
    tensor::Tensor,
};

use super::empty_strided_cpu;

fn to_impl(self_: &Tensor, options: &TensorOptions, non_blocking: bool, copy: bool) -> Tensor {
    let mut memory_format = options
        .memory_format_opt()
        .unwrap_or(MemoryFormat::Preserve);

    if self_.dtype() == options.dtype()
        && self_.layout() == options.layout()
        && self_.device() == options.device()
        && !copy
        && (memory_format == MemoryFormat::Preserve
            || self_.suggest_memory_format(false) == memory_format)
    {
        return self_.clone();
    }

    if memory_format == MemoryFormat::Preserve {
        if self_.is_non_overlapping_and_dense() {
            // Copy all strides
            let r = empty_strided_cpu(self_.sizes(), self_.strides(), options);
            r.copy(self_, Some(non_blocking));
            return r;
        } else {
            memory_format = self_.suggest_memory_format(false);
        }
    }
    // See Note [Explicit nullopt MemoryFormat argument]
    let r = empty(
        self_.sizes(),
        options.set_memory_format(Some(memory_format)),
        None,
    );
    r.copy(self_, Some(non_blocking));
    r
}

/// Defaults:
///    non_blocking: false
///    copy: false
///    optional_memory_format: None
pub fn to_dtype(
    self_: &Tensor,
    dtype: ScalarType,
    non_blocking: bool,
    copy: bool,
    optional_memory_format: Option<MemoryFormat>,
) -> Tensor {
    to_impl(
        self_,
        &self_.options().set_dtype_(dtype),
        non_blocking,
        copy,
    )
}
