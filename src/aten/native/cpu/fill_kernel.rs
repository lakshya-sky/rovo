use crate::aten::native::loops;
use crate::tensor::NewTensorIterator;

pub fn fill_kernel(mut iter: NewTensorIterator, value: f32) {
    let op = move || value;
    loops::cpu_kernel_vec(&mut iter, op);
}
