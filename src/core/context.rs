use super::*;
struct Context;

impl Context {
    pub fn default_generator(&self, device: Device) -> Generator<impl GeneratorImpl> {
        match device {
            super::Device::CPU => get_default_cpu_generator(),
            super::Device::CUDA => todo!(),
        }
    }
}

#[inline]
pub fn init() {
    global_context();
}

fn global_context() -> Context {
    todo!()
}
pub fn manual_seed(seed: usize) {
    let context = global_context();
    let gen = context.default_generator(Device::CPU);
    {
        gen.set_current_seed(seed);
    }
}
