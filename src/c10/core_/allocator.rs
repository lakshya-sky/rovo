use super::{Device, DeviceType, UniqueVoidPtr};
use std::ffi::c_void;
use std::ptr::NonNull;
pub struct DataPtr {
    ptr: UniqueVoidPtr,
    device: Device,
}
impl Default for DataPtr {
    fn default() -> Self {
        Self {
            ptr: UniqueVoidPtr::default(),
            device: Device::default(),
        }
    }
}

impl DataPtr {
    pub fn new(data: Option<NonNull<c_void>>) -> Self {
        Self {
            ptr: UniqueVoidPtr::new(data),
            ..Self::default()
        }
    }
    pub fn new_with_deleter(data: Option<NonNull<c_void>>, deleter: fn(*mut c_void)) -> Self {
        Self {
            ptr: UniqueVoidPtr::new_with_deleter(data, deleter),
            ..Self::default()
        }
    }
    pub fn release_context(self) -> Option<*mut c_void> {
        self.ptr.release_context()
    }

    pub fn get(&self) -> Option<&NonNull<c_void>> {
        self.ptr.get()
    }

    pub fn clear(self) {
        self.ptr.clear()
    }
    pub fn device(&self) -> Device {
        self.device.clone()
    }
    pub fn is_empty(&self) -> bool {
        self.ptr.is_empty()
    }
}

pub trait Allocator {
    fn allocate(&self, n: usize) -> DataPtr;
    // Not sure right now, how to return nullptr.
    // fn raw_deleter();

    fn raw_allocate(&self, n: usize) -> Option<*mut c_void> {
        let dptr = self.allocate(n);
        dptr.release_context()
    }
}

pub struct AllocatorRegisterer;

static mut ALLOCATORARRAY: [Option<*mut dyn Allocator>;
    DeviceType::COMPILE_TIME_MAX_DEVICE_TYPE as usize] =
    [None; DeviceType::COMPILE_TIME_MAX_DEVICE_TYPE as usize];
static mut ALLOCATORPRIORITY: [usize; DeviceType::COMPILE_TIME_MAX_DEVICE_TYPE as usize] =
    [0; DeviceType::COMPILE_TIME_MAX_DEVICE_TYPE as usize];

pub fn reigster_allocator(alloc: *mut dyn Allocator, t: DeviceType) {
    set_allocator(t, alloc, 0);
}

fn set_allocator(t: DeviceType, alloc: *mut dyn Allocator, priority: usize) {
    unsafe {
        if priority >= ALLOCATORPRIORITY[t as usize] {
            ALLOCATORARRAY[t as usize] = Some(alloc);
            ALLOCATORPRIORITY[t as usize] = priority;
        }
    }
}
pub fn get_allocator(t: DeviceType) -> *mut dyn Allocator {
    unsafe {
        let alloc = ALLOCATORARRAY[t as usize];
        assert!(alloc.is_some(), "Allocator for {:?} is not set", t);
        alloc.unwrap()
    }
}
