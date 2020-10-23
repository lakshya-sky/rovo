use super::mt19937_engine;
use super::{Generator, GeneratorImpl};
use std::ops::Shl;
use std::sync::{Arc, Mutex};

#[inline]
fn make_64_bits_from_32_bits(hi: u32, lo: u32) -> u64 {
    (hi as u64).shl(32) | lo as u64
}

// Note: Pytorch uses mutex field in subclass of GeneratorImpl class
// to explitly acquire lock. But rust needs every thing be `impl Send`
// for allowing to send it across threads. Hence I ensure that lock
// is acquired implicitly. For example: pytorch first locks mutex
// and then calls methods such as random and set_current_seed. while here
// random and set_current_seed internally locks engine to mutate change.
// I didn't know an alternative and propaly is related to how rust works compare
// to C++. Note [Acquire lock when using random generators] in pytorch documentation
// is related to this same behavior.
pub struct CPUGeneratorImpl {
    engine: Arc<Mutex<mt19937_engine::MT19937Engine>>,
    next_float_normal_sample: Option<f32>,
    next_double_normal_sample: Option<f64>,
}

impl CPUGeneratorImpl {
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            engine: Arc::new(Mutex::new(mt19937_engine::MT19937Engine::new(match seed {
                Some(s) => Some(s),
                None => Some(super::DEFAULT_RNG_SEED_VAL),
            }))),
            next_float_normal_sample: None,
            next_double_normal_sample: None,
        }
    }

    pub fn seed(&mut self) -> u64 {
        let random = super::get_non_deterministic_random(false);
        self.set_current_seed(random);
        random
    }

    pub fn device_type(&self) -> super::super::Device {
        super::super::Device::CPU
    }

    pub fn random_(&self) -> u32 {
        self.engine.lock().unwrap().call()
    }

    pub fn random64_(&self) -> u64 {
        let mut engine = self.engine.lock().unwrap();
        let random1 = engine.call();
        let random2 = engine.call();
        make_64_bits_from_32_bits(random1, random2)
    }

    pub fn set_engine(&mut self, engine: Arc<Mutex<mt19937_engine::MT19937Engine>>) {
        self.engine = engine;
    }

    pub fn as_self(&self) -> Self {
        let mut gen = Self::new(None);
        gen.set_engine(self.engine.clone());
        gen.set_next_float_normal_sample(self.next_float_normal_sample);
        gen.set_next_double_normal_sample(self.next_double_normal_sample);
        gen
    }
}

impl GeneratorImpl for CPUGeneratorImpl {
    fn random(&self) -> u32 {
        self.random_()
    }

    fn set_current_seed(&mut self, seed: u64) {
        let mut engine = self.engine.lock().unwrap();
        self.next_double_normal_sample = None;
        self.next_float_normal_sample = None;
        *engine = mt19937_engine::MT19937Engine::new(Some(seed));
        // self.engine = Arc::new(Mutex::new(mt19937_engine::MT19937Engine::new(Some(seed))))
    }

    fn current_seed(&self) -> u64 {
        self.engine.lock().unwrap().seed()
    }

    // `clone_impl` clones this impl with a new engine and a different Arc.
    // while `clone`  calls clone on the Arc providing same engine.
    fn clone_impl(&self) -> Box<dyn GeneratorImpl> {
        let mut gen = Self::new(None);
        gen.set_engine(Arc::new(Mutex::new(self.engine.lock().unwrap().clone())));
        gen.set_next_float_normal_sample(self.next_float_normal_sample);
        gen.set_next_double_normal_sample(self.next_double_normal_sample);
        Box::new(gen)
    }

    fn random64(&self) -> u64 {
        self.random64_()
    }

    fn as_cpu_impl(&self) -> CPUGeneratorImpl {
        self.as_self()
    }

    fn set_next_float_normal_sample(&mut self, next_float: Option<f32>) {
        self.next_float_normal_sample = next_float
    }

    fn set_next_double_normal_sample(&mut self, next_double: Option<f64>) {
        self.next_double_normal_sample = next_double
    }
    fn next_float_normal_sample(&self) -> Option<f32> {
        self.next_float_normal_sample
    }

    fn next_double_normal_sample(&self) -> Option<f64> {
        self.next_double_normal_sample
    }

    fn clone(&self) -> Box<dyn GeneratorImpl> {
        let mut gen = Self::new(None);
        gen.set_engine(self.engine.clone());
        gen.set_next_float_normal_sample(self.next_float_normal_sample);
        gen.set_next_double_normal_sample(self.next_double_normal_sample);
        Box::new(gen)
    }
}

static mut DEFAULT_CPU_GEN: Option<Generator> = None;
pub fn get_default_cpu_generator() -> &'static mut Generator {
    unsafe {
        if DEFAULT_CPU_GEN.is_none() {
            DEFAULT_CPU_GEN = Some(create_cpu_generator(Some(
                super::get_non_deterministic_random(false),
            )));
        }
        DEFAULT_CPU_GEN.as_mut().unwrap()
    }
}

pub fn create_cpu_generator(seed: Option<u64>) -> Generator {
    super::make_generator(Box::new(CPUGeneratorImpl::new(seed)))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::core::*;
    use crate::tensor::*;
    use std::mem::MaybeUninit;

    #[derive(Clone)]
    struct TestCPUGenerator {
        value: u64,
        next_float_normal_sample: Option<f32>,
        next_double_normal_sample: Option<f64>,
    }

    impl TestCPUGenerator {
        pub fn new(value: u64) -> Self {
            Self {
                value,
                next_double_normal_sample: None,
                next_float_normal_sample: None,
            }
        }
    }

    impl GeneratorImpl for TestCPUGenerator {
        fn random(&self) -> u32 {
            self.value as u32
        }

        fn set_current_seed(&mut self, _seed: u64) {
            todo!()
        }

        fn current_seed(&self) -> u64 {
            todo!()
        }

        fn clone_impl(&self) -> Box<dyn GeneratorImpl> {
            todo!()
        }

        fn random64(&self) -> u64 {
            self.value as u64
        }

        fn as_cpu_impl(&self) -> CPUGeneratorImpl {
            todo!()
        }

        fn next_float_normal_sample(&self) -> Option<f32> {
            self.next_float_normal_sample
        }

        fn set_next_float_normal_sample(&mut self, randn: Option<f32>) {
            self.next_float_normal_sample = randn;
        }

        fn next_double_normal_sample(&self) -> Option<f64> {
            self.next_double_normal_sample
        }

        fn set_next_double_normal_sample(&mut self, randn: Option<f64>) {
            self.next_double_normal_sample = randn;
        }

        fn clone(&self) -> Box<dyn GeneratorImpl> {
            todo!()
        }
    }

    const MAGIC_NUMBER: u64 = 424242424242424242;

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

    struct NormalDistribution<T> {
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
            let r: T = (num::cast::<_, T>(-2.0).unwrap()
                * (num::cast::<_, T>(1.0).unwrap() - u2).ln())
            .sqrt();
            let theta: T = num::cast::<_, T>(2.0 * std::f64::consts::PI).unwrap() * u1;
            if type_name == "f64" {
                maybe_set_next_double_normal_sample(gen, r * theta.sin())
            } else {
                maybe_set_next_float_normal_sample(gen, r * theta.sin())
            }
            let ret = r * theta.cos();
            let result = transformation_helper::normal(ret, self.mean, self.std);
            result
        }
    }

    pub fn normal_kernel(self_: &Tensor, mean: f64, std: f64, gen: &mut dyn GeneratorImpl) {
        let iter = crate::tensor::TensorIterator::nullary_op(self_);
        let closure = move || {
            let normal = NormalDistribution::new(mean, std);
            normal.call(gen)
        };
        crate::aten::native::cpu_serial_kernel(iter, closure);
    }

    struct NormalKernel;

    impl NormalKernel {
        pub fn call(&self, self_: &Tensor, mean: f64, std: f64, gen: Option<&mut Generator>) {
            normal_kernel(self_, mean, std, check_generator(gen.unwrap()))
        }
    }

    fn normal<'a>(
        tensor: &'a Tensor,
        mean: f64,
        std: f64,
        gen: Option<&mut Generator>,
    ) -> &'a Tensor {
        let normal_kernel = NormalKernel;
        normal_impl_(normal_kernel, tensor, mean, std, gen)
    }

    fn normal_impl_<'a>(
        kernel: NormalKernel,
        self_: &'a Tensor,
        mean: f64,
        std: f64,
        gen: Option<&mut Generator>,
    ) -> &'a Tensor {
        kernel.call(self_, mean, std, gen);
        self_
    }

    #[test]
    fn rngtest_normal() {
        let mean = 123.45;
        let std = 67.89;
        let mut gen = super::super::make_generator(Box::new(TestCPUGenerator::new(MAGIC_NUMBER)));
        let actual = Tensor::empty(&[10]);
        let _ = normal(&actual, mean, std, Some(&mut gen));
        let expected = Tensor::empty_like(&actual);
        normal_kernel(&expected, mean, std, check_generator(&mut gen));

        // Todo: Needs to vefiry that both tensor are equal or close to equal by tolerance.
        // Function similar to below should be used to achive the purpose.
        // assert!(all_close(&actual, &expected));
        // The skeleton is available in aten::native::tensor_compare.
    }
}
