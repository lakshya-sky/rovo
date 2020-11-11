use crate::aten::native::loops;
use crate::c10::{Scalar, ScalarType};
use crate::tensor::NewTensorIterator;
use crate::Closure;

pub fn fill_kernel(mut iter: NewTensorIterator, value: Scalar) {
    let scalar_t = iter.dtype();
    match scalar_t {
        ScalarType::Float => {
            let value = value.to::<f32>();
            let op = move |_args: [f32; 0]| value;
            let closure = Closure::new(op);
            loops::cpu_kernel_vec(&mut iter, closure);
        }
        ScalarType::Int => {
            let value = value.to::<i32>();
            let op = move |_args: [f32; 0]| value;
            let closure = Closure::new(op);
            loops::cpu_kernel_vec(&mut iter, closure);
        }
        ScalarType::Double => {
            let value = value.to::<f64>();
            let op = move |_args: [f32; 0]| value;
            let closure = Closure::new(op);
            loops::cpu_kernel_vec(&mut iter, closure);
        }
        ScalarType::Long => {
            let value = value.to::<i64>();
            let op = move |_args: [f32; 0]| value;
            let closure = Closure::new(op);
            loops::cpu_kernel_vec(&mut iter, closure);
        }
        _ => todo!(),
    }
}
