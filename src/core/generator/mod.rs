use std::fs::File;
use std::io::Read;
use std::primitive::u64;

mod mt19937_engine;
#[derive(Clone)]
pub struct Generator<T: GeneratorImpl> {
    impl_: T,
}

impl<T: GeneratorImpl> Generator<T> {
    pub fn set_current_seed(&self, _seed: usize) {
        todo!()
    }

    pub fn new(gen_impl: T) -> Self {
        Self { impl_: gen_impl }
    }
}

const DEFAULT_RNG_SEED_VAL: u64 = 67280421310721;
pub trait GeneratorImpl {
    fn random(&mut self) -> u32;
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

pub fn check_generator<'a, T: 'a + GeneratorImpl>(gen: &'a mut Generator<T>) -> &'a mut T {
    &mut gen.impl_
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_read_urandom() {
        println!("{}", read_random_long());
    }

    #[test]
    fn test_cloning() {
        let mut gen1 = create_cpu_generator(None);
        {
            let cpu_gen1 = check_generator::<CPUGeneratorImpl>(&mut gen1);
            cpu_gen1.random();
            cpu_gen1.random();
        }
        let mut gen2 = gen1.clone();
        let cpu_gen2 = check_generator::<CPUGeneratorImpl>(&mut gen2);
        let cpu_gen1 = check_generator::<CPUGeneratorImpl>(&mut gen1);
        assert_eq!(cpu_gen1.random(), cpu_gen2.random());
    }
}
