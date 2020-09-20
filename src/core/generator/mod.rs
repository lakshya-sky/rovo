use std::fs::File;
use std::io::Read;
use std::primitive::u64;
use std::sync::{Arc, Mutex};
mod mt19937_engine;

pub struct Generator<T: GeneratorImpl> {
    impl_: Arc<Mutex<T>>,
}

impl<T: GeneratorImpl> Clone for Generator<T> {
    fn clone(&self) -> Self {
        let self_locked = self.impl_.lock().unwrap();
        let clone = self_locked.clone();
        Generator::new(clone)
    }
}

impl<T: GeneratorImpl> Generator<T> {
    pub fn new(gen_impl: T) -> Self {
        Self {
            impl_: Arc::new(Mutex::new(gen_impl)),
        }
    }
}

const DEFAULT_RNG_SEED_VAL: u64 = 67280421310721;
pub trait GeneratorImpl {
    fn random(&mut self) -> u32;
    fn clone(&self) -> Self;
    fn set_current_seed(&mut self, seed: u64);
    fn current_seed(&self) -> u64;
    // fn clone_impl(&mut self) -> u32;
}

mod cpu_generator;
pub use cpu_generator::*;

pub fn make_generator<T: GeneratorImpl>(impl_: T) -> Generator<T> {
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

pub fn check_generator<T: GeneratorImpl>(gen: &Generator<T>) -> Arc<Mutex<T>> {
    gen.impl_.clone()
}

#[cfg(test)]
mod test {
    use super::*;
    use std::thread;
    #[test]
    fn test_read_urandom() {
        println!("{}", read_random_long());
    }

    #[test]
    fn test_cloning() {
        let mut gen1 = create_cpu_generator(None);

        let cpu_gen1 = check_generator::<CPUGeneratorImpl>(&mut gen1);

        {
            let mut cpu_gen1_locked = cpu_gen1.lock().unwrap();
            println!("{}", cpu_gen1_locked.random());
            println!("{}", cpu_gen1_locked.random());
        }

        let mut gen2 = gen1.clone();
        let cpu_gen2 = check_generator::<CPUGeneratorImpl>(&mut gen2);
        let mut cpu_gen2_locked = cpu_gen2.lock().unwrap();
        let mut cpu_gen1_locked = cpu_gen1.lock().unwrap();

        assert_eq!(cpu_gen1_locked.random(), cpu_gen2_locked.random());
    }

    #[test]
    fn test_multithreading_get_engine_operator() {
        let gen1 = create_cpu_generator(None);
        let cpu_gen1 = check_generator::<CPUGeneratorImpl>(&gen1);
        let gen2;
        {
            gen2 = gen1.clone();
        }

        let cpu_gen1_clone1 = cpu_gen1.clone();
        let handle1 = thread::spawn(move || {
            let mut impl_ = cpu_gen1_clone1.lock().unwrap();
            impl_.random();
        });
        let cpu_gen1_clone2 = cpu_gen1.clone();
        let handle2 = thread::spawn(move || {
            let mut impl_ = cpu_gen1_clone2.lock().unwrap();
            impl_.random();
        });
        let cpu_gen1_clone3 = cpu_gen1.clone();
        let handle3 = thread::spawn(move || {
            let mut impl_ = cpu_gen1_clone3.lock().unwrap();
            impl_.random();
        });

        let _ = handle1.join();
        let _ = handle2.join();
        let _ = handle3.join();

        let cpu_gen2 = check_generator::<CPUGeneratorImpl>(&gen2);
        cpu_gen2.lock().unwrap().random();
        cpu_gen2.lock().unwrap().random();
        cpu_gen2.lock().unwrap().random();
        assert_eq!(
            cpu_gen1.lock().unwrap().random(),
            cpu_gen2.lock().unwrap().random()
        );
    }

    #[test]
    fn test_set_get_current_seed() {
        let foo = get_default_cpu_generator();
        let mut impl_ = foo.impl_.lock().unwrap();
        impl_.set_current_seed(123);
        let current_seed = impl_.current_seed();
        assert_eq!(current_seed, 123);
    }

    #[test]
    fn test_multithreading_get_set_current_seed() {
        let gen1 = get_default_cpu_generator();
        let initial_seed;
        {
            initial_seed = gen1.impl_.lock().unwrap().current_seed();
        }
        let cpu_gen = check_generator(&gen1);

        let cpu_gen_clone1 = cpu_gen.clone();
        let handle1 = thread::spawn(move || {
            let mut impl_ = cpu_gen_clone1.lock().unwrap();
            let current_seed = impl_.current_seed();
            impl_.set_current_seed(current_seed + 1);
        });

        let cpu_gen_clone2 = cpu_gen.clone();
        let handle2 = thread::spawn(move || {
            let mut impl_ = cpu_gen_clone2.lock().unwrap();
            let current_seed = impl_.current_seed();
            impl_.set_current_seed(current_seed + 1);
        });

        let _ = handle1.join();
        let _ = handle2.join();
        let impl_ = cpu_gen.lock().unwrap();
        assert_eq!(initial_seed + 2, impl_.current_seed())
    }
}
