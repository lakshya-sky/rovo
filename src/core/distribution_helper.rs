use super::{generator::GeneratorImpl, transformation_helper};
use std::mem::MaybeUninit;
pub struct UniformRealDistribution<T> {
    from: T,
    to: T,
}
impl<T> UniformRealDistribution<T>
where
    T: Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Mul<Output = T>
        + num::cast::NumCast,
{
    pub fn new(from: T, to: T) -> Self {
        Self { from, to }
    }
    pub fn call(&self, gen: &dyn GeneratorImpl) -> T {
        if std::any::type_name::<T>() == "f64" {
            transformation_helper::uniform_real(gen.random64(), self.from, self.to)
        } else {
            transformation_helper::uniform_real(gen.random(), self.from, self.to)
        }
    }
}
pub struct NormalDistribution<T> {
    mean: T,
    std: T,
}

impl<T> NormalDistribution<T>
where
    T: std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + From<f32>
        + Copy
        + std::ops::Sub<Output = T>
        + num::cast::NumCast
        + num_traits::Float,
{
    pub fn new(mean: T, std: T) -> Self {
        Self { mean, std }
    }

    pub fn call(&self, gen: &mut dyn GeneratorImpl) -> T {
        let mut ret = MaybeUninit::<T>::uninit();
        let type_name = std::any::type_name::<T>();
        if type_name == "f64" {
            if maybe_get_next_double_normal_sample(gen, ret.as_mut_ptr()) {
                let ret = unsafe { ret.assume_init() };
                return transformation_helper::normal(ret, self.mean, self.std);
            }
        } else {
            if maybe_get_next_float_normal_sample(gen, ret.as_mut_ptr()) {
                let ret = unsafe { ret.assume_init() };
                return transformation_helper::normal(ret, self.mean, self.std);
            }
        }
        let uniform = UniformRealDistribution::<T>::new(0.0.into(), 1.0.into());
        // 0.10037074167330773
        let u1 = uniform.call(gen);
        let u2 = uniform.call(gen);
        // 0.45994029116614604
        let r: T =
            (num::cast::<_, T>(-2.0).unwrap() * (num::cast::<_, T>(1.0).unwrap() - u2).ln()).sqrt();
        let theta: T = num::cast::<_, T>(2.0 * std::f64::consts::PI).unwrap() * u1;
        if type_name == "f64" {
            maybe_set_next_double_normal_sample(gen, r * theta.sin())
        } else {
            maybe_set_next_float_normal_sample(gen, r * theta.sin())
        }
        let ret = r * theta.cos();
        let result = transformation_helper::normal(ret, self.mean, self.std);
        // 148.66901331974375
        result
    }
}
pub fn maybe_get_next_double_normal_sample<T: num::cast::NumCast>(
    gen: &mut dyn GeneratorImpl,
    ret: *mut T,
) -> bool {
    if let Some(rt) = gen.next_double_normal_sample() {
        unsafe {
            *ret = num::cast(rt).unwrap();
        }
        gen.set_next_double_normal_sample(None);
        return true;
    }
    false
}

pub fn maybe_get_next_float_normal_sample<T: num::cast::NumCast>(
    gen: &mut dyn GeneratorImpl,
    ret: *mut T,
) -> bool {
    if let Some(rt) = gen.next_float_normal_sample() {
        unsafe {
            *ret = num::cast(rt).unwrap();
        }
        gen.set_next_float_normal_sample(None);
        return true;
    }
    false
}

pub fn maybe_set_next_double_normal_sample<T: num::cast::NumCast>(
    gen: &mut dyn GeneratorImpl,
    ret: T,
) {
    gen.set_next_double_normal_sample(Some(num::cast(ret).unwrap()));
}

pub fn maybe_set_next_float_normal_sample<T: num::cast::NumCast>(
    gen: &mut dyn GeneratorImpl,
    ret: T,
) {
    gen.set_next_float_normal_sample(Some(num::cast(ret).unwrap()));
}
