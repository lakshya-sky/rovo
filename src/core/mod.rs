mod grad_mode;
pub use grad_mode::*;

mod generator;
pub use generator::*;

mod context;
pub use context::*;

pub enum Device {
    CPU,
    CUDA,
}
