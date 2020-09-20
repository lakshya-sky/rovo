use super::mt19937_engine;
use super::{Generator, GeneratorImpl};
use std::ops::Shl;

#[inline]
fn make_64_bits_from_32_bits(hi: u32, lo: u32) -> u64 {
    (hi as u64).shl(32) | lo as u64
}
pub struct CPUGeneratorImpl {
    engine: mt19937_engine::MT19937Engine,
    next_float_normal_sample: Option<f32>,
    next_double_normal_sample: Option<f64>,
}

impl CPUGeneratorImpl {
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            engine: mt19937_engine::MT19937Engine::new(match seed {
                Some(s) => Some(s),
                None => Some(super::DEFAULT_RNG_SEED_VAL),
            }),
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

    pub fn random_(&mut self) -> u32 {
        self.engine.call()
    }

    pub fn random64_(&mut self) -> u64 {
        let random1 = self.engine.call();
        let random2 = self.engine.call();
        make_64_bits_from_32_bits(random1, random2)
    }

    pub fn set_engine(&mut self, engine: mt19937_engine::MT19937Engine) {
        self.engine = engine;
    }

    pub fn set_next_float_normal_sample(&mut self, next_float: Option<f32>) {
        self.next_float_normal_sample = next_float
    }

    pub fn set_next_double_normal_sample(&mut self, next_double: Option<f64>) {
        self.next_double_normal_sample = next_double
    }

    pub fn clone_impl(&self) -> Self {
        let mut gen = Self::new(None);
        gen.set_engine(self.engine);
        gen.set_next_float_normal_sample(self.next_float_normal_sample);
        gen.set_next_double_normal_sample(self.next_double_normal_sample);
        gen
    }
}

// impl Clone for CPUGeneratorImpl {
//     fn clone(&self) -> Self {
//         self.clone_impl()
//     }
// }

impl GeneratorImpl for CPUGeneratorImpl {
    fn random(&mut self) -> u32 {
        self.random_()
    }

    fn clone(&self) -> Self {
        self.clone_impl()
    }

    fn set_current_seed(&mut self, seed: u64) {
        self.next_double_normal_sample = None;
        self.next_float_normal_sample = None;
        self.engine = mt19937_engine::MT19937Engine::new(Some(seed))
    }

    fn current_seed(&self) -> u64 {
        self.engine.seed()
    }
}

pub fn get_default_cpu_generator() -> Generator<CPUGeneratorImpl> {
    let default_gen_cpu = create_cpu_generator(Some(super::get_non_deterministic_random(false)));
    default_gen_cpu
}

pub fn create_cpu_generator(seed: Option<u64>) -> Generator<CPUGeneratorImpl> {
    super::make_generator(CPUGeneratorImpl::new(seed))
}
