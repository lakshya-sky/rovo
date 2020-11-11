mod device_type;
pub use device_type::*;

mod unique_void_ptr;
pub use unique_void_ptr::*;

mod allocator;
pub use allocator::*;

mod cpu_allocator;
pub use cpu_allocator::*;

mod storage;
pub use storage::*;

mod layout;
pub use layout::*;

mod memory_format;
pub use memory_format::*;

mod scalar_type;
pub use scalar_type::*;

mod scalar;
pub use scalar::*;

mod tensor_options;
pub use tensor_options::*;

pub fn allocator_init() {
    cpu_allocator::register_cpu_allocator();
}
