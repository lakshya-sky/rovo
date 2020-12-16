use crate::c10::{DeviceType, KCPU};

use super::*;
struct Context;

impl Context {
    pub fn default_generator(&self, device: DeviceType) -> &'static mut Generator {
        match device {
            DeviceType::CPU => get_default_cpu_generator(),
            _ => todo!(),
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
    let gen = context.default_generator(KCPU);
    gen.set_current_seed(seed);
}

pub fn get_cpu_allocator() -> *mut dyn crate::c10::Allocator {
    crate::c10::get_cpu_allocator()
}
