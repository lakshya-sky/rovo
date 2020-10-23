use crate::core::{GeneratorImpl, UniformRealDistribution};
use crate::tensor::TensorIterator;

pub fn uniform_kernel(iter: TensorIterator, from: f64, to: f64, gen: &mut dyn GeneratorImpl) {
    let uniform = UniformRealDistribution::new(from, to);

    let closure = move || uniform.call(gen);

    crate::aten::native::cpu_serial_kernel(iter, closure);
}
