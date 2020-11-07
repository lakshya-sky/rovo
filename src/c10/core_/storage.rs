use super::*;
use once_cell::sync::OnceCell;
use std::cell::RefCell;
use std::mem;
use std::ptr::NonNull;
use std::rc::Rc;
pub struct StorageImpl {
    data_ptr: DataPtr,
    size_bytes: usize,
    resizable: bool,
    received_cuda: bool,
    allocator: Option<*mut dyn Allocator>,
}

impl StorageImpl {
    pub fn new(
        size_bytes: usize,
        data_ptr: DataPtr,
        allocator: *mut dyn Allocator,
        resizable: bool,
    ) -> Self {
        if resizable {
            assert!(
                !allocator.is_null(),
                "For resizable storage, allocator must be provided"
            )
        }
        Self {
            size_bytes,
            data_ptr,
            resizable,
            allocator: Some(allocator),
            received_cuda: false,
        }
    }
    pub fn undefined_instance() -> Self {
        Self {
            data_ptr: DataPtr::default(),
            size_bytes: 0,
            received_cuda: false,
            resizable: false,
            allocator: None,
        }
    }
    pub fn reset(self) {
        self.data_ptr.clear();
    }
    pub fn data<T>(&self) -> Option<*mut T> {
        let data = self.data_ptr.get();
        match data {
            Some(p) => Some(p.as_ptr() as *mut T),
            None => None,
        }
    }
    pub fn nbytes(&self) -> usize {
        self.size_bytes
    }
    pub fn set_nbytes(&mut self, size_bytes: usize) {
        self.size_bytes = size_bytes
    }
    pub fn release_resources(self) {
        self.data_ptr.clear()
    }
    pub fn resizable(&self) -> bool {
        self.resizable
    }
    pub fn data_ptr(&self) -> &DataPtr {
        &self.data_ptr
    }
    pub fn set_data_ptr(&mut self, mut data_ptr: DataPtr) -> DataPtr {
        mem::swap(&mut self.data_ptr, &mut data_ptr);
        data_ptr
    }
    pub fn allocator(&self) -> &mut dyn Allocator {
        unsafe { &mut *self.allocator.unwrap() }
    }
    pub fn device(&self) -> Device {
        self.data_ptr.device()
    }
}

#[derive(Clone)]
pub struct Storage {
    storage_impl: Rc<RefCell<StorageImpl>>,
}
unsafe impl Sync for Storage {}
unsafe impl Send for Storage {}

static SINGLETON: OnceCell<Storage> = OnceCell::new();

impl Default for Storage {
    fn default() -> Self {
        let impl_ =
            SINGLETON.get_or_init(|| Self::new_from_impl(StorageImpl::undefined_instance()));
        impl_.clone()
    }
}

impl Storage {
    pub fn new_from_impl(impl_: StorageImpl) -> Self {
        Self {
            storage_impl: Rc::new(RefCell::new(impl_)),
        }
    }

    pub fn new(ptr: Rc<RefCell<StorageImpl>>) -> Self {
        Self { storage_impl: ptr }
    }
    pub fn data<T>(&self) -> NonNull<T> {
        match self.get_unsafe_storage_impl().data() {
            Some(d) => unsafe { NonNull::new_unchecked(d) },
            None => panic!("data ptr is null"),
        }
    }
    pub fn unsafe_data<T>(&self) -> NonNull<T> {
        self.data()
    }

    pub fn resizable(&self) -> bool {
        self.storage_impl.borrow().resizable()
    }
    pub fn nbytes(&self) -> usize {
        self.storage_impl.borrow().nbytes()
    }
    pub fn data_ptr(&self) -> &DataPtr {
        let tmp = self.storage_impl.as_ptr();
        unsafe { &(*tmp).data_ptr() }
    }
    pub fn set_data_ptr(&self, data_ptr: DataPtr) -> DataPtr {
        self.storage_impl.borrow_mut().set_data_ptr(data_ptr)
    }
    pub fn allocator<'a>(&'a self) -> &'a mut dyn Allocator {
        let impl_ = self.get_unsafe_storage_impl();
        (&mut *impl_).allocator()
    }
    pub fn unsafe_release_storage_impl(self) -> *mut StorageImpl {
        let result = self.storage_impl.as_ptr();
        std::mem::forget(self.storage_impl);
        result
    }
    pub fn get_unsafe_storage_impl(&self) -> &mut StorageImpl {
        unsafe { &mut *self.storage_impl.as_ptr() }
    }
    pub fn use_count(&self) -> usize {
        Rc::strong_count(&self.storage_impl)
    }
    pub fn unique(&self) -> bool {
        Rc::strong_count(&self.storage_impl) == 1
    }
    pub fn device(&self) -> Device {
        self.storage_impl.borrow().device()
    }
}

impl std::convert::From<StorageImpl> for Storage {
    fn from(item: StorageImpl) -> Self {
        Storage::new(Rc::new(RefCell::new(item)))
    }
}
impl std::convert::From<&Self> for Storage {
    fn from(item: &Self) -> Self {
        item.clone()
    }
}
