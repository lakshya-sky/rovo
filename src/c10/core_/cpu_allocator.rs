use super::*;
use std::ffi::c_void;
use std::ptr::NonNull;

const GALIGNMENT: usize = 64;
pub struct DefaultCPUAllocator;

impl Allocator for DefaultCPUAllocator {
    fn allocate(&self, nbytes: usize) -> DataPtr {
        let data = alloc_cpu(nbytes);
        DataPtr::new_with_deleter(data, report_and_delete)
    }
}
fn alloc_cpu(nbytes: usize) -> Option<NonNull<c_void>> {
    if nbytes == 0 {
        return None;
    }
    unsafe {
        let mut data: *mut c_void = std::mem::MaybeUninit::uninit().assume_init();
        let err = libc::posix_memalign(&mut data, GALIGNMENT, nbytes);
        if err != 0 {
            panic!(format!(
                "DefaultCPUAllocator: can't allocate memory: you tried to allocate {} bytes. Error code {}({})",
                nbytes,
                err,
                std::ffi::CStr::from_ptr(libc::strerror(err) as *const libc::c_char)
                    .to_str()
                    .unwrap()
            ));
        }
        Some(NonNull::new(data).unwrap())
    }
}

fn report_and_delete(ptr_: *mut c_void) {
    if ptr_.is_null() {
        return;
    }
    free_cpu(ptr_);
}

fn free_cpu(data: *mut c_void) {
    unsafe { libc::free(data) }
}

pub static mut GCPUALLOC: DefaultCPUAllocator = DefaultCPUAllocator;

pub fn get_default_cpu_allocator() -> *mut DefaultCPUAllocator {
    unsafe { &mut GCPUALLOC as *mut DefaultCPUAllocator }
}

pub fn register_cpu_allocator() {
    reigster_allocator(get_default_cpu_allocator(), DeviceType::CPU)
}

pub fn get_cpu_allocator() -> *mut dyn Allocator {
    get_allocator(DeviceType::CPU)
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_cpu_allocation_deallocation() {
        unsafe {
            crate::init_rovo();
            let allocator = crate::c10::get_allocator(DeviceType::CPU);
            {
                let _ = (&*allocator).allocate(4);
            }
        }
    }
    #[test]
    fn test_allocation_assignment() {
        unsafe {
            crate::init_rovo();
            let allocator = crate::c10::get_allocator(DeviceType::CPU);
            {
                let data = (&*allocator).allocate(4);
                let ptr = data.get();
                match ptr {
                    Some(p) => {
                        let m = p.as_ptr();
                        (m as *mut u32).write(51);
                    }
                    _ => {}
                }
                assert_eq!(*(ptr.unwrap().as_ptr() as *mut u32), 51);
            }
        }
    }
}
