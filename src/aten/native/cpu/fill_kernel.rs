use crate::aten::native::loops;
use crate::tensor::NewTensorIterator;
use crate::Closure;

pub fn fill_kernel(mut iter: NewTensorIterator, value: f32) {
    let op = move |_args: [f32; 0]| value;
    let closure = Closure::new(op);
    loops::cpu_kernel_vec(&mut iter, closure);
}
