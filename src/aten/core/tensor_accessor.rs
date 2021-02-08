use std::{
    marker::PhantomData,
    ops::{Index, IndexMut},
    sync::atomic::{AtomicPtr, Ordering::Relaxed},
};

pub trait Number {
    const VALUE: usize;
}

pub struct One;

pub struct PlusOne<T> {
    _marker: PhantomData<T>,
}

impl Number for One {
    const VALUE: usize = 1;
}

impl<T: Number> Number for PlusOne<T> {
    const VALUE: usize = <T as Number>::VALUE + 1;
}
pub trait TensorAccessorTrait<T> {
    type Outut;
    fn len(&self) -> usize;
    fn sizes(&self) -> &[usize];
    fn size(&self, i: usize) -> usize;
    fn strides(&self) -> &[usize];
    fn stride(&self, i: usize) -> usize;
    fn data_mut(&self) -> &AtomicPtr<T>;
    //fn index(&self, i: usize) -> Self::Outut;
}

struct ConstPtr(*const usize);
unsafe impl Send for ConstPtr {}
unsafe impl Sync for ConstPtr {}

pub struct TensorAccessor<T, N: Number> {
    sizes: ConstPtr,
    strides: ConstPtr,
    data: AtomicPtr<T>,
    len: usize,
    _number: PhantomData<N>,
}

impl<T, N: Number> TensorAccessor<T, N> {
    pub fn new(data: *mut T, sizes: *const usize, strides: *const usize, len: usize) -> Self {
        Self {
            data: data.into(),
            sizes: ConstPtr(sizes),
            strides: ConstPtr(strides),
            len,
            _number: PhantomData,
        }
    }
}

impl<T, N: Number> TensorAccessor<T, PlusOne<N>> {
    pub fn index(&self, index: usize) -> TensorAccessor<T, N> {
        let (data, sizes, strides) = unsafe {
            let data = self.data.load(Relaxed).add(self.stride(0) * index);
            let sizes = self.sizes.0.add(1);
            let strides = (self.strides.0).add(1);
            (data, sizes, strides)
        };
        let accessor = TensorAccessor::new(data, sizes, strides, self.len - 1);
        accessor
    }
}

impl<T, N: Number> TensorAccessorTrait<T> for TensorAccessor<T, PlusOne<N>> {
    type Outut = TensorAccessor<T, N>;

    fn len(&self) -> usize {
        self.len
    }

    fn sizes(&self) -> &[usize] {
        unsafe { std::slice::from_raw_parts(self.sizes.0, self.len) }
    }

    fn size(&self, i: usize) -> usize {
        unsafe { self.sizes.0.add(i).read() }
    }

    fn strides(&self) -> &[usize] {
        unsafe { std::slice::from_raw_parts(self.strides.0, self.len) }
    }

    fn stride(&self, i: usize) -> usize {
        unsafe { self.strides.0.add(i).read() }
    }

    fn data_mut(&self) -> &AtomicPtr<T> {
        &self.data
    }

    // fn index(&self, index: usize) -> Self::Outut {
    //     let (data, sizes, strides) = unsafe {
    //         let data = self.data.load(Relaxed).add(self.stride(0) * index);
    //         let sizes = self.sizes.0.add(1);
    //         let strides = (self.strides.0).add(1);
    //         (data, sizes, strides)
    //     };
    //     let accessor = TensorAccessor::new(data, sizes, strides, self.len - 1);
    //     accessor
    // }
}

impl<T> TensorAccessorTrait<T> for TensorAccessor<T, One> {
    type Outut = T;

    fn len(&self) -> usize {
        self.len
    }

    fn sizes(&self) -> &[usize] {
        unsafe { std::slice::from_raw_parts(self.sizes.0, self.len) }
    }

    fn size(&self, i: usize) -> usize {
        unsafe { self.sizes.0.add(i).read() }
    }

    fn strides(&self) -> &[usize] {
        unsafe { std::slice::from_raw_parts(self.strides.0, self.len) }
    }

    fn stride(&self, i: usize) -> usize {
        unsafe { self.strides.0.add(i).read() }
    }

    fn data_mut(&self) -> &AtomicPtr<T> {
        &self.data
    }

    // fn index(&self, index: usize) -> Self::Outut {
    //     let data = unsafe {
    //         let data = self.data_mut().load(Relaxed).add(self.stride(0) * index);
    //         data.read()
    //     };
    //     data
    // }
}

impl<T> Index<usize> for TensorAccessor<T, One> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        let data = unsafe {
            let data = self.data_mut().load(Relaxed).add(self.stride(0) * index);
            &*data
        };
        data
    }
}

impl<T> IndexMut<usize> for TensorAccessor<T, One> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let data = unsafe {
            let data = self.data_mut().load(Relaxed).add(self.stride(0) * index);
            &mut *data
        };
        data
    }
}

pub struct SingleDimensionalTensorAccessor<T> {
    sizes: ConstPtr,
    strides: ConstPtr,
    data: AtomicPtr<T>,
    len: usize,
}

impl<T> SingleDimensionalTensorAccessor<T> {
    pub fn new(data: *mut T, sizes: *const usize, strides: *const usize) -> Self {
        Self {
            data: data.into(),
            sizes: ConstPtr(sizes),
            strides: ConstPtr(strides),
            len: 1,
        }
    }
    pub fn insert_at(&self, index: usize, item: T) {
        unsafe {
            let data = self.data_mut().load(Relaxed).add(self.stride(0) * index);
            data.write(item)
        };
    }
    fn len(&self) -> usize {
        self.len
    }

    fn sizes(&self) -> &[usize] {
        unsafe { std::slice::from_raw_parts(self.sizes.0, self.len) }
    }

    fn size(&self, i: usize) -> usize {
        unsafe { self.sizes.0.add(i).read() }
    }

    fn strides(&self) -> &[usize] {
        unsafe { std::slice::from_raw_parts(self.strides.0, self.len) }
    }

    fn stride(&self, i: usize) -> usize {
        unsafe { self.strides.0.add(i).read() }
    }

    fn data_mut(&self) -> &AtomicPtr<T> {
        &self.data
    }

    // fn index(&self, index: usize) -> Self::Output {
    //     let data = unsafe {
    //         let data = self.data_mut().add(self.stride(0) * index);
    //         data.read()
    //     };
    //     data
    // }
    fn index(&self, index: usize) -> T {
        let data = unsafe {
            let data = self.data_mut().load(Relaxed).add(self.stride(0) * index);
            data.read()
        };
        data
    }
}
