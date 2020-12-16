use crate::{
    autograd::empty,
    c10::TensorOptions,
    c10::{Device, MemoryFormat, ScalarType},
    tensor::Tensor,
};

use super::empty_strided_cpu;

// Take a Device that may not have device_index set (i.e., having it as -1
// representing the current device) and return the corresponding Device
// according to the actual device at the time of this function call.  No-op
// if the device_index is set.
#[inline(always)]
fn ensure_has_index(device: Device) -> Device {
    if device.is_cpu() || device.has_index() {
        return device;
    }
    // Pytorch uses DeviceGuardImplInterface here
    Device::default()
}

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
    _optional_memory_format: Option<MemoryFormat>,
) -> Tensor {
    to_impl(
        self_,
        &self_.options().set_dtype_(dtype),
        non_blocking,
        copy,
    )
}

/// Defaults:
///    non_blocking: false
///    copy: false
///    optional_memory_format: None
pub fn to<O: AsRef<TensorOptions>>(
    self_: &Tensor,
    options: O,
    non_blocking: bool,
    copy: bool,
    optional_memory_format: Option<MemoryFormat>,
) -> Tensor {
    let options = options.as_ref();
    assert!(
        !(options.has_memory_format() && optional_memory_format.is_some()),
        "Cannot set memory_format both in TensorOptions and explicit argument; please delete the redundant setter.");
    let mut options =
        options.merge_in(TensorOptions::default().set_memory_format(optional_memory_format));

    assert!(
        options.requires_grad_opt() == None,
        "to(options) expects unset requires_grad flag, but got options.requires_grad set as {}",
        options.requires_grad()
    );

    assert!(!options.has_layout() || self_.layout() == options.layout(),
               "to(options) doesn't support converting to a different layout, but got self.layout being {:?} and options.layout set as {:?}", self_.layout(),
                options.layout());

    if options.has_device() {
        options = options.set_device(ensure_has_index(options.device()));
    }
    let specified_options = self_.options().merge_in(options);
    to_impl(self_, &specified_options, non_blocking, copy)
}
