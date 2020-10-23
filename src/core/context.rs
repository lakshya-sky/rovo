use super::*;
struct Context;

impl Context {
    pub fn default_generator(&self, device: Device) -> &'static mut Generator {
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

static GLOBAL_CONTEXT: Context = Context {};

fn global_context() -> &'static Context {
    &GLOBAL_CONTEXT
}

pub fn manual_seed(seed: u64) {
    let context = global_context();
    let gen = context.default_generator(Device::CPU);
    gen.set_current_seed(seed);
}

pub fn getCPUAllocator()->*mut dyn crate::c10::Allocator{
    crate::c10::get_cpu_allocator()
}