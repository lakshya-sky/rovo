use super::AutogradMeta;
use crate::c10::{MemoryFormat, TensorOptions};
use crate::tensor::{Edge, Tensor, TensorVersion};
use crate::{aten::native, rsrc::TensorDataContainer};
use std::rc::Rc;

// default arg:
// requires_grad : false
pub fn make_variable(data: Tensor, requires_grad: bool) -> Tensor {
    if data.defined() {
        if Rc::strong_count(&data._impl) == 1 && data.get_unsafe_tensor_impl().unique_version() {
            let mut data_impl = Rc::try_unwrap(data._impl).unwrap().into_inner().unwrap();
            if requires_grad {
                data_impl.set_autograd_meta(Some(AutogradMeta::new_without_edge(
                    &data_impl,
                    requires_grad,
                )));
            } else {
                data_impl.set_autograd_meta(None);
            }
            return Tensor::from_impl(data_impl);
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
            return Tensor::from_impl(other_impl_copy);
        }
    }
    Tensor::default()
}

pub fn make_variable_with_edge(other: Tensor, gradient_edge: Edge) -> Tensor {
    if other.defined() {
        let mut other_impl_copy = other
            .get_unsafe_tensor_impl()
            .shallow_copy_and_detach(other.get_unsafe_tensor_impl().version_counter());
        other_impl_copy.set_autograd_meta(Some(AutogradMeta::new(
            &other_impl_copy,
            false,
            gradient_edge,
        )));
        Tensor::from_impl(other_impl_copy)
    } else {
        Tensor::default()
    }
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
) -> Tensor {
    let options = get_options(options);
    let tensor = native::empty(size, options.set_requires_grad(None), memory_format);
    let result = make_variable(tensor, options.requires_grad());
    result
}

pub fn empty_like<A: Into<Option<TensorOptions>>, M: Into<Option<MemoryFormat>>>(
    self_: &Tensor,
    options: A,
    memory_format: M,
) -> Tensor {
    let options = get_options(options);
    let tensor = native::empty_like(self_, options.set_requires_grad(None), memory_format);
    let result = make_variable(tensor, options.requires_grad());
    result
}

pub fn ones<A: Into<Option<TensorOptions>>>(size: &[usize], options: A) -> Tensor {
    let options = get_options(options);
    let tensor = native::ones(size, options.set_requires_grad(None));
    let result = make_variable(tensor, options.requires_grad());
    result
}

pub fn ones_like<A: Into<Option<TensorOptions>>>(self_: &Tensor, options: A) -> Tensor {
    let options = get_options(options);
    let tensor = native::ones_like(self_, options.set_requires_grad(None), None);
    let result = make_variable(tensor, options.requires_grad());
    result
}

pub fn full<A: Into<Option<TensorOptions>>>(size: &[usize], fill_value: f32, options: A) -> Tensor {
    let options = get_options(options);
    let tensor = native::full(size, fill_value, options.set_requires_grad(None));
    let result = make_variable(tensor, options.requires_grad());
    result
}

pub fn full_like<A: Into<Option<TensorOptions>>>(
    self_: &Tensor,
    fill_value: f32,
    options: A,
) -> Tensor {
    let options = get_options(options);
    let tensor = native::full_like(self_, fill_value, options.set_requires_grad(None), None);
    let result = make_variable(tensor, options.requires_grad());
    result
}

pub fn tensor<A: Into<Option<TensorOptions>>>(
    tensor_data_container: impl Into<TensorDataContainer>,
    options: A,
) -> Tensor {
    let options = get_options(options);
    let tensor_data_container: TensorDataContainer = tensor_data_container.into();
    let result = make_variable(
        tensor_data_container.convert_to_tensor(options.set_requires_grad(None)),
        options.requires_grad(),
    );
    result
}
