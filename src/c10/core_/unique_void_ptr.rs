use std::ffi::c_void;
use std::ptr::NonNull;
extern crate libc;

pub type Deleter<T> = fn(*mut T);

fn delete_nothing<T>(_ptr: *mut T) {}
fn default_deleter<T>(ptr_: *mut T) {
    unsafe {
        let _ = Box::from_raw(ptr_);
    }
}

pub struct Unique<T> {
    inner: Option<NonNull<T>>,
    deleter: Deleter<T>,
}

impl Default for Unique<c_void> {
    fn default() -> Self {
        Self {
            inner: None,
            deleter: delete_nothing,
        }
    }
}

impl<T> Drop for Unique<T> {
    fn drop(&mut self) {
        if self.inner.is_some() {
            (self.deleter)(self.inner.unwrap().as_ptr())
        }
    }
}

impl<T> Unique<T> {
    pub fn new(inner: Option<NonNull<T>>) -> Self {
        Self {
            inner,
            deleter: delete_nothing,
        }
    }
    pub fn new_with_deleter(inner: Option<NonNull<T>>, deleter: Deleter<T>) -> Self {
        Self { inner, deleter }
    }
    pub fn release_context(self) -> Option<*mut T> {
        let result = self.inner;
        std::mem::forget(self);
        result.map_or(None, |x| Some(x.as_ptr()))
    }

    pub fn clear(self) {
        std::mem::drop(self);
    }
    pub fn get(&self) -> Option<&NonNull<T>> {
        self.inner.as_ref()
    }
    pub fn is_empty(&self) -> bool {
        self.inner.is_some()
    }
}

pub type UniqueVoidPtr = Unique<c_void>;

#[cfg(test)]
mod test {
    use super::*;
    use libc::posix_memalign;
    use std::mem::MaybeUninit;
    #[test]
    fn test_allocation() {
        unsafe {
            let mut x = MaybeUninit::uninit().assume_init();
            let _ = posix_memalign(&mut x, 64, 4);
            (x as *mut u32).write(32);
            {
                let _ = Unique::new_with_deleter(Some(NonNull::new_unchecked(x)), default_deleter);
            }
        }
    }
}
