use crate::aten::native::cpu;
use crate::aten::util;
use crate::core::{get_default_cpu_generator, Generator};
use crate::tensor::TensorIterator;

pub fn uniform_kernel(iter: TensorIterator, from: f64, to: f64, mut gen: Option<Generator>) {
    let default_gen = &mut get_default_cpu_generator();
    let generator = util::get_generator_or_default(gen.as_mut(), default_gen);
    cpu::distribution_templates::uniform_kernel(iter, from, to, generator);
}
