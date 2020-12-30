use std::ops::Index;

/*
pub struct TensorAccessorBase<T, const N: usize> {
    sizes: *const usize,
    strides: *const usize,
    data: *mut T,
}

impl<T, const N: usize> TensorAccessorBase<T, N> {
    pub fn new(data: *mut T, sizes: *const usize, strides: *const usize) -> Self {
        Self {
            data,
            sizes,
            strides,
        }
    }
    pub fn sizes(&self) -> &[usize] {
        unsafe { std::slice::from_raw_parts(self.sizes, N) }
    }
    pub fn size(&self, i: usize) -> usize {
        unsafe { self.sizes.add(i).read() }
    }
    pub fn strides(&self) -> &[usize] {
        unsafe { std::slice::from_raw_parts(self.strides, N) }
    }
    pub fn stride(&self, i: usize) -> usize {
        unsafe { self.strides.add(i).read() }
    }

    pub fn data_mut(&self) -> *mut T {
        self.data
    }

    pub fn data(&self) -> *const T {
        self.data as *const T
    }
}
*/
pub trait TensorAccessor<T> {
    fn sizes(&self) -> &[usize];
    fn size(&self, i: usize) -> usize;
    fn strides(&self) -> &[usize];
    fn stride(&self, i: usize) -> usize;
    fn data_mut(&self) -> *mut T;
    fn data(&self) -> *const T;
}
struct CPUTensorAccessor<T> {
    sizes: *const usize,
    strides: *const usize,
    data: *mut T,
    len: usize,
}

impl<T> CPUTensorAccessor<T> {
    pub fn new(data: *mut T, sizes: *const usize, strides: *const usize, len: usize) -> Self {
        Self {
            data,
            sizes,
            strides,
            len,
        }
    }

    pub fn get(&self, index: usize) -> Self {
        let (data, sizes, strides) = unsafe {
            let data = self.data.add(self.stride(0) * index);
            let sizes = self.sizes.add(1);
            let strides = self.strides.add(1);
            (data, sizes, strides)
        };
        let accessor = CPUTensorAccessor::new(data, sizes, strides, self.len - 1);
        accessor
    }
}

impl<T> TensorAccessor<T> for CPUTensorAccessor<T> {
    fn sizes(&self) -> &[usize] {
        unsafe { std::slice::from_raw_parts(self.sizes, self.len) }
    }

    fn size(&self, i: usize) -> usize {
        unsafe { self.sizes.add(i).read() }
    }

    fn strides(&self) -> &[usize] {
        unsafe { std::slice::from_raw_parts(self.strides, self.len) }
    }

    fn stride(&self, i: usize) -> usize {
        unsafe { self.strides.add(i).read() }
    }

    fn data_mut(&self) -> *mut T {
        self.data
    }

    fn data(&self) -> *const T {
        self.data as *const T
    }
}
