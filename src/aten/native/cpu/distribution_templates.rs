use crate::c10::ScalarType;
use crate::core::*;
use crate::tensor::{Tensor, TensorIterator};
use crate::Closure;

pub fn uniform_kernel(
    mut iter: TensorIterator,
    from: f64,
    to: f64,
    gen: &mut dyn GeneratorImpl,
) {
    let uniform = UniformRealDistribution::new(from, to);
    let closure = Closure::new(|_args: [f64; 0]| uniform.call(gen));

    crate::aten::native::cpu_serial_kernel(&mut iter, closure);
}
pub struct NormalKernel;

impl NormalKernel {
    pub fn call(&self, self_: &Tensor, mean: f64, std: f64, gen: Option<&mut Generator>) {
        normal_kernel(self_, mean, std, check_generator(gen.unwrap()))
    }
}

pub fn normal_kernel(self_: &Tensor, mean: f64, std: f64, gen: &mut dyn GeneratorImpl) {
    let size = self_.numel();
    if self_.scalar_type() == ScalarType::Float && size >= 16 {
        normal_fill(self_, mean as f32, std as f32, gen);
    } else {
        let the_type = self_.scalar_type();
        match the_type {
            ScalarType::Byte => {}
            ScalarType::Int => {}
            ScalarType::Float => {
                type SCALAR = f32;
                if size >= 16 && self_.is_contiguous() {
                    normal_fill(self_, mean as SCALAR, std as SCALAR, gen);
                } else {
                    let mut iter = crate::tensor::TensorIterator::nullary_op(self_);

                    let closure = Closure::new(|_args: [f64; 0]| {
                        let normal = NormalDistribution::new(mean, std);
                        normal.call(gen) as SCALAR
                    });
                    crate::aten::native::cpu_serial_kernel(&mut iter, closure);
                }
            }
            ScalarType::Double => {}
            _ => {}
        }
        //  const auto& the_type = self.scalar_type();
        //   at::ScalarType _st = ::detail::scalar_type(the_type);
        //   switch (_st) {
        //     case at::ScalarType::Double: {
        //       using scalar_t = double;
        //       return [&] {
        //         if (size >= 16 && self.is_contiguous()) {
        //           normal_fill<scalar_t>(
        //               self,
        //               static_cast<scalar_t>(mean),
        //               static_cast<scalar_t>(std),
        //               generator);
        //         } else {
        //           auto iter = TensorIterator::nullary_op(self);
        //           std::lock_guard<std::mutex> lock(generator->mutex_);
        //           cpu_serial_kernel(iter, [mean, std, generator]() -> scalar_t {
        //             at::normal_distribution<double> normal(mean, std);
        //             return static_cast<scalar_t>(normal(generator));
        //           });
        //         }
        //       }();
        //     }
        //     case at::ScalarType::Float: {
        //       using scalar_t = float;
        //       return [&] {
        //         if (size >= 16 && self.is_contiguous()) {
        //           normal_fill<scalar_t>(
        //               self,
        //               static_cast<scalar_t>(mean),
        //               static_cast<scalar_t>(std),
        //               generator);
        //         } else {
        //           auto iter = TensorIterator::nullary_op(self);
        //           std::lock_guard<std::mutex> lock(generator->mutex_);
        //           cpu_serial_kernel(iter, [mean, std, generator]() -> scalar_t {
        //             at::normal_distribution<double> normal(mean, std);
        //             return static_cast<scalar_t>(normal(generator));
        //           });
        //         }
        //       }();
        //     }
    }
}

fn normal_fill<T>(_self_: &Tensor, _mean: T, _std: T, _gen: &mut dyn GeneratorImpl) {
    todo!()
}
