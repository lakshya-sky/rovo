use crate::aten::native;
use crate::autograd::AutogradMeta;
use crate::c10::{MemoryFormat, TensorOptions};
use crate::tensor::{Edge, NewTensor, TensorVersion};
use std::rc::Rc;

pub fn make_variable(data: NewTensor, requires_grad: bool) -> NewTensor {
    if data.defined() {
        if Rc::strong_count(&data._impl) == 1 && data.get_unsafe_tensor_impl().unique_version() {
            let mut data_impl = Rc::try_unwrap(data._impl).unwrap().into_inner();
            if requires_grad {
                data_impl.set_autograd_meta(Some(AutogradMeta::new_without_edge(
                    &data_impl,
                    requires_grad,
                )));
            } else {
                data_impl.set_autograd_meta(None);
            }
            return NewTensor::from_impl(data_impl);
        } else {
            let mut other_impl_copy = data
                .get_unsafe_tensor_impl()
                .shallow_copy_and_detach(&TensorVersion::new());
            if requires_grad {
                other_impl_copy.set_autograd_meta(Some(AutogradMeta::new_without_edge(
                    &other_impl_copy,
                    requires_grad,
                )))
            } else {
                other_impl_copy.set_autograd_meta(None);
            }
            return NewTensor::from_impl(other_impl_copy);
        }
    }
    NewTensor::default()
}

pub fn make_variable_with_edge(other: NewTensor, gradient_edge: Edge) -> NewTensor {
    let mut other_impl_copy = other
        .get_unsafe_tensor_impl()
        .shallow_copy_and_detach(other.get_unsafe_tensor_impl().version_counter());
    other_impl_copy.set_autograd_meta(Some(AutogradMeta::new(
        &other_impl_copy,
        false,
        gradient_edge,
    )));
    NewTensor::from_impl(other_impl_copy)
}

pub fn get_options<A: Into<Option<TensorOptions>>>(options: A) -> TensorOptions {
    match options.into() {
        Some(o) => o,
        None => TensorOptions::default(),
    }
}

pub fn empty<A: Into<Option<TensorOptions>>, M: Into<Option<MemoryFormat>>>(
    size: &[usize],
    options: A,
    memory_format: M,
) -> NewTensor {
    let options = get_options(options);
    let tensor =
        (|| -> NewTensor { native::empty(size, options.set_requires_grad(None), memory_format) })();
    let result = make_variable(tensor, options.requires_grad());
    result
}

pub fn empty_like<A: Into<Option<TensorOptions>>, M: Into<Option<MemoryFormat>>>(
    self_: &NewTensor,
    options: A,
    memory_format: M,
) -> NewTensor {
    let options = get_options(options);
    let tensor = (|| -> NewTensor {
        native::empty_like(self_, options.set_requires_grad(None), memory_format)
    })();
    let result = make_variable(tensor, options.requires_grad());
    result
}

pub fn ones<A: Into<Option<TensorOptions>>>(size: &[usize], options: A) -> NewTensor {
    let options = get_options(options);
    let tensor = (|| -> NewTensor { native::ones(size, options.set_requires_grad(None)) })();
    let result = make_variable(tensor, options.requires_grad());
    result
}

pub fn ones_like<A: Into<Option<TensorOptions>>>(self_: &NewTensor, options: A) -> NewTensor {
    let options = get_options(options);
    let tensor =
        (|| -> NewTensor { native::ones_like(self_, options.set_requires_grad(None), None) })();
    let result = make_variable(tensor, options.requires_grad());
    result
}

pub fn full<A: Into<Option<TensorOptions>>>(
    size: &[usize],
    fill_value: f32,
    options: A,
) -> NewTensor {
    let options = get_options(options);
    let tensor =
        (|| -> NewTensor { native::full(size, fill_value, options.set_requires_grad(None)) })();
    let result = make_variable(tensor, options.requires_grad());
    result
}

pub fn full_like<A: Into<Option<TensorOptions>>>(
    self_: &NewTensor,
    fill_value: f32,
    options: A,
) -> NewTensor {
    let options = get_options(options);
    let tensor = (|| -> NewTensor {
        native::full_like(self_, fill_value, options.set_requires_grad(None), None)
    })();
    let result = make_variable(tensor, options.requires_grad());
    result
}
