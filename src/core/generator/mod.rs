use std::fs::File;
use std::io::Read;
use std::primitive::u64;

pub mod mt19937_engine;

pub struct Generator {
    impl_: Box<dyn GeneratorImpl>,
}

impl Clone for Generator {
    fn clone(&self) -> Self {
        Generator::new(self.impl_.clone_impl())
    }
}

impl Generator {
    pub fn new(gen_impl: Box<dyn GeneratorImpl>) -> Self {
        Self { impl_: gen_impl }
    }

    pub fn set_current_seed(&mut self, seed: u64) {
        self.impl_.set_current_seed(seed)
    }

    pub fn current_seed(&self) -> u64 {
        self.impl_.current_seed()
    }

    pub fn as_with_cpu_impl(&self) -> Generator {
        Generator::new(Box::new(self.impl_.as_cpu_impl()))
    }
}

const DEFAULT_RNG_SEED_VAL: u64 = 67280421310721;
pub trait GeneratorImpl: Send {
    fn random(&self) -> u32;
    fn random64(&self) -> u64;
    fn set_current_seed(&mut self, seed: u64);
    fn current_seed(&self) -> u64;
    fn clone_impl(&self) -> Box<dyn GeneratorImpl>;
    fn clone(&self) -> Box<dyn GeneratorImpl>;
    fn as_cpu_impl(&self) -> CPUGeneratorImpl;
    fn next_float_normal_sample(&self) -> Option<f32>;
    fn set_next_float_normal_sample(&mut self, randn: Option<f32>);
    fn next_double_normal_sample(&self) -> Option<f64>;
    fn set_next_double_normal_sample(&mut self, randn: Option<f64>);
}

mod cpu_generator;
pub use cpu_generator::*;

pub fn make_generator(impl_: Box<dyn GeneratorImpl>) -> Generator {
    Generator::new(impl_)
}

pub fn get_non_deterministic_random(is_cuda: bool) -> u64 {
    let s: u64;
    if !is_cuda {
        s = read_random_long();
    } else {
        // Todo: pytorch Implementation

        // `std::random_device rd;
        // limit to 53 bits to ensure unique representation in double
        // s = ((((uint64_t)rd()) << 32) + rd()) & 0x1FFFFFFFFFFFFF;`
        s = 111111;
    }
    s
}

pub fn read_random_long() -> u64 {
    let mut rand_dev = File::open("/dev/urandom").unwrap();
    let mut buf = [0u8; 8];
    rand_dev.read(&mut buf).unwrap();
    let random_number = u64::from_le_bytes(buf);
    random_number
}

pub fn check_generator(gen: &mut Generator) -> &mut dyn GeneratorImpl {
    let t = gen.impl_.as_mut();
    t
}

#[cfg(test)]
mod test {
    use super::*;
    use std::thread;
    #[test]
    fn test_read_urandom() {
        println!(
            "Read Random Value from /dev/urandom: {}",
            read_random_long()
        );
    }

    #[test]
    fn test_cloning() {
        let mut gen1 = create_cpu_generator(None);
        {
            let cpu_gen1 = check_generator(&mut gen1);
            cpu_gen1.random();
            cpu_gen1.random();
        }

        let mut gen2 = gen1.clone();
        let cpu_gen2 = check_generator(&mut gen2);
        let cpu_gen1 = check_generator(&mut gen1);
        assert_eq!(cpu_gen1.random(), cpu_gen2.random());
    }

    #[test]
    fn test_multithreading_get_engine_operator() {
        let mut gen1 = create_cpu_generator(None);

        let mut gen2 = gen1.clone();

        let cpu_gen1 = check_generator(&mut gen1);

        let cpu_gen1_clone1 = cpu_gen1.clone();
        let handle1 = thread::spawn(move || {
            cpu_gen1_clone1.random();
        });

        let cpu_gen1_clone2 = cpu_gen1.clone();
        let handle2 = thread::spawn(move || {
            cpu_gen1_clone2.random();
        });

        let cpu_gen1_clone3 = cpu_gen1.clone();

        let handle3 = thread::spawn(move || {
            cpu_gen1_clone3.random();
        });

        let _ = handle1.join();
        let _ = handle2.join();
        let _ = handle3.join();

        let cpu_gen2 = check_generator(&mut gen2);
        cpu_gen2.random();
        cpu_gen2.random();
        cpu_gen2.random();

        let left = cpu_gen1.random();
        let right = cpu_gen2.random();
        assert_eq!(left, right);
    }

    #[test]
    fn test_set_get_current_seed() {
        let foo = get_default_cpu_generator();
        foo.set_current_seed(123);
        let current_seed = foo.current_seed();
        assert_eq!(current_seed, 123);
    }

    #[test]
    fn test_multithreading_get_set_current_seed() {
        let mut gen1 = get_default_cpu_generator();
        let initial_seed = gen1.current_seed();
        let cpu_gen = check_generator(&mut gen1);

        let mut cpu_gen_clone1 = cpu_gen.clone();
        let handle1 = thread::spawn(move || {
            // let mut impl_ = cpu_gen_clone1.lock().unwrap();
            let current_seed = cpu_gen_clone1.current_seed();
            cpu_gen_clone1.set_current_seed(current_seed + 1);
        });

        let mut cpu_gen_clone2 = cpu_gen.clone();
        let handle2 = thread::spawn(move || {
            // let mut impl_ = cpu_gen_clone2.lock().unwrap();
            let current_seed = cpu_gen_clone2.current_seed();
            cpu_gen_clone2.set_current_seed(current_seed + 1);
        });

        let _ = handle1.join();
        let _ = handle2.join();
        // let impl_ = cpu_gen.lock().unwrap();
        assert_eq!(initial_seed + 2, cpu_gen.current_seed())
    }
}
