use crate::aten::native::{cpu, loops};
use crate::aten::util;
use crate::core::{get_default_cpu_generator, Generator};
use crate::tensor::{Tensor, TensorIterator};
use crate::Closure;
use crate::AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2;

pub fn uniform_kernel(iter: TensorIterator, from: f64, to: f64, mut gen: Option<Generator>) {
    let default_gen = &mut get_default_cpu_generator();
    let generator = util::get_generator_or_default(gen.as_mut(), default_gen);
    cpu::distribution_templates::uniform_kernel(iter, from, to, generator);
}

pub fn normal_kernel(self_: &Tensor, mean: f64, std: f64, mut gen: Option<Generator>) {
    let default_gen = &mut get_default_cpu_generator();
    let generator = util::get_generator_or_default(gen.as_mut(), default_gen);
    cpu::distribution_templates::normal_kernel(self_, mean, std, generator);
}

pub fn sigmoid_kernel(iter: &mut TensorIterator) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2!(_, _, iter.dtype(), "sigmoid_cpu", || {
        loops::cpu_kernel_vec(
            iter,
            Closure::new(|args: [Scalart; 1]| -> Scalart {
                let a = args[0];
                let exp_a: Scalart = f32::exp(-a as f32) as Scalart;
                let one: Scalart = 1 as Scalart;
                one / (one + exp_a)
            }),
        )
    })
}

pub fn neg_kernel(iter: &mut TensorIterator) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2!(_, _, iter.dtype(), "neg_cpu", || {
        loops::cpu_kernel_vec(
            iter,
            Closure::new(|args: [Scalart; 1]| -> Scalart { -args[0] }),
        )
    })
}
