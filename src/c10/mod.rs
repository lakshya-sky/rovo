mod core_;
pub use core_::*;

mod util;
pub use util::*;

pub fn init(){
    core_::allocator_init();
}