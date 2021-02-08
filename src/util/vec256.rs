use num::Float;
use std::{
    ffi::c_void,
    iter::FromIterator,
    marker::PhantomData,
    ops::{Index, IndexMut, Mul, Sub},
};

use std::ops::Add;

/// PyTorch uses array of 256 bit internally for Vec256.
/// I am assuming that the array lives on stack rather than heap.
/// Since I can't find any other way to create that array at compile time.
/// I am using const N for this.
/// The contraint for that is N = 32/size_of(T)
#[derive(Debug, Clone)]
pub struct Vec256<T: Float> {
    buf: [u8; 32],
    el_size: usize,
    _ph: PhantomData<T>,
}

impl<T> Vec256<T>
where
    T: Float + Sized,
{
    pub fn new() -> Self {
        Self {
            buf: [0; 32],
            el_size: std::mem::size_of::<T>(),
            _ph: PhantomData,
        }
    }

    pub fn filled_new(val: T) -> Self {
        let v = Self::new().into_iter().map(|_i| val).collect();
        v
    }

    pub fn size() -> usize {
        32 / std::mem::size_of::<T>()
    }

    #[inline(always)]
    pub fn loadu<C: Into<Option<usize>>>(ptr: *const c_void, count: C) -> Self {
        let mut v = Self::new();
        let count = match count.into() {
            Some(c) => c * std::mem::size_of::<T>(),
            None => 32,
        };
        let dst = v.buf.as_mut_ptr() as *mut c_void;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, dst, count);
        }
        v
    }

    #[inline(always)]
    pub fn store<C: Into<Option<usize>>>(&self, ptr: *mut c_void, count: C) {
        let count = match count.into() {
            Some(c) => c,
            None => Self::size(),
        };

        let src = self.buf.as_ptr() as *const c_void;
        unsafe {
            std::ptr::copy_nonoverlapping(src, ptr, count * self.el_size);
        }
    }

    #[inline(always)]
    pub fn set<C: Into<Option<usize>>>(a: &Self, b: &Self, count: C) -> Self {
        let count = match count.into() {
            Some(c) => c,
            None => Self::size(),
        };
        let mut vec = Self::new();
        for i in 0..Self::size() {
            if i < count {
                vec[i] = b[i];
            } else {
                vec[i] = a[i];
            }
        }
        vec
    }

    // It should be able to handle case where a single element might be nan.
    #[inline(always)]
    pub fn maximum(a: &Vec256<T>, b: &Vec256<T>) -> Vec256<T> {
        let mut c = Vec256::new();
        for i in 0..Vec256::<T>::size() {
            c[i] = if a[i] > b[i] { a[i] } else { b[i] };
        }
        c
    }
    #[inline(always)]
    pub fn exp(&self) -> Self {
        self.into_iter().map(|e| e.exp()).collect()
    }

    #[inline(always)]
    pub fn log(&self) -> Self {
        self.into_iter().map(|e| e.ln()).collect()
    }
}

pub struct Vec256IntoIterator<T: num::Float> {
    vec: Vec256<T>,
    index: usize,
}

impl<T: num::Float> IntoIterator for Vec256<T> {
    type Item = T;

    type IntoIter = Vec256IntoIterator<T>;

    fn into_iter(self) -> Self::IntoIter {
        Vec256IntoIterator {
            index: 0,
            vec: self,
        }
    }
}

impl<T: num::Float> Iterator for Vec256IntoIterator<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < Vec256::<T>::size() {
            let index = self.index * std::mem::size_of::<T>();
            self.index += 1;
            Some(unsafe { *(self.vec.buf.as_ptr().offset(index as isize) as *const T) })
        } else {
            None
        }
    }
}
pub struct Vec256IntoIteratorRef<'a, T: num::Float> {
    vec: &'a Vec256<T>,
    index: usize,
}

impl<'a, T: num::Float> IntoIterator for &'a Vec256<T> {
    type Item = T;

    type IntoIter = Vec256IntoIteratorRef<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        Vec256IntoIteratorRef {
            index: 0,
            vec: self,
        }
    }
}

impl<'a, T: num::Float> Iterator for Vec256IntoIteratorRef<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < Vec256::<T>::size() {
            let index = self.index * std::mem::size_of::<T>();
            self.index += 1;
            Some(unsafe { *(self.vec.buf.as_ptr().offset(index as isize) as *const T) })
        } else {
            None
        }
    }
}

impl<T: num::Float> FromIterator<T> for Vec256<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut c = Self::new();
        let mut idx = 0;
        for i in iter {
            c[idx] = i;
            idx += 1;
        }
        c
    }
}

impl<T: num::Float> Index<usize> for Vec256<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        if index < Self::size() {
            let index = index * std::mem::size_of::<T>();
            unsafe { &*(self.buf.as_ptr().offset(index as isize) as *const T) }
        } else {
            panic!("Index out of range")
        }
    }
}

impl<T: num::Float> IndexMut<usize> for Vec256<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index < Self::size() {
            let index = index * std::mem::size_of::<T>();
            unsafe { &mut *(self.buf.as_ptr().offset(index as isize) as *mut T) }
        } else {
            panic!("Index out of range")
        }
    }
}

impl<T: num::Float> Add<Self> for Vec256<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.into_iter()
            .zip(rhs.into_iter())
            .map(|(a, b)| a + b)
            .collect()
    }
}

impl<T: num::Float> Sub<Self> for Vec256<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.into_iter()
            .zip(rhs.into_iter())
            .map(|(a, b)| a - b)
            .collect()
    }
}

impl<T: num::Float> Mul<Self> for Vec256<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.into_iter()
            .zip(rhs.into_iter())
            .map(|(a, b)| a * b)
            .collect()
    }
}

// TODO: Make this more efficient
#[inline(always)]
pub unsafe fn vec_reduce_all<T: num::Float, F>(
    mut vec_fun: F,
    mut acc_vec: Vec256<T>,
    size: usize,
) -> T
where
    F: FnMut(Vec256<T>, Vec256<T>) -> Vec256<T>,
{
    let _vec_size = Vec256::<T>::size();
    let el_size = acc_vec.el_size;
    let mut acc_arr = [0u8; 32];
    acc_vec.store(acc_arr.as_mut_ptr() as *mut c_void, None);
    for i in 1..size {
        let mut acc_arr_next = [0u8; 32];

        let src = acc_arr.as_ptr().offset((i * el_size) as isize) as *const T;
        let dst = acc_arr_next.as_mut_ptr() as *mut T;
        std::ptr::copy_nonoverlapping(src, dst, 1);
        let acc_vec_next = Vec256::loadu(dst as *const c_void, None);
        acc_vec = vec_fun(acc_vec, acc_vec_next);
    }
    acc_vec.store(acc_arr.as_mut_ptr() as *mut c_void, None);
    return *(acc_arr.as_ptr() as *const T);
}

#[inline(always)]
pub fn reduce_all<T: num::Float, F>(mut vec_fun: F, data: *mut T, size: usize) -> T
where
    F: FnMut(Vec256<T>, Vec256<T>) -> Vec256<T>,
{
    let vec_size = Vec256::<T>::size();
    if size < vec_size {
        unsafe { return vec_reduce_all(vec_fun, Vec256::loadu(data as *const c_void, size), size) }
    }
    let mut acc_vec = Vec256::loadu(data as *const c_void, None);
    let mut d = vec_size;
    let limit = size - (size % vec_size);
    while d < limit {
        let data_vec = Vec256::loadu(unsafe { data.offset(d as isize) as *const c_void }, None);
        acc_vec = vec_fun(acc_vec, data_vec);
        d += vec_size;
    }

    if size - d > 0 {
        let data_vec = Vec256::loadu(
            unsafe { data.offset(d as isize) as *const c_void },
            size - d,
        );
        let tmp = &vec_fun(acc_vec.clone(), data_vec);
        acc_vec = Vec256::set(&acc_vec, tmp, size - d);
    }
    unsafe { vec_reduce_all(vec_fun, acc_vec, vec_size) }
}

#[inline(always)]
pub fn map_reduce_all<T: num::Float, M, R>(
    mut map_fun: M,
    mut red_fun: R,
    data: *mut T,
    size: usize,
) -> T
where
    M: FnMut(Vec256<T>) -> Vec256<T>,
    R: FnMut(Vec256<T>, Vec256<T>) -> Vec256<T>,
{
    let vec_size = Vec256::<T>::size();
    if size < vec_size {
        return unsafe {
            vec_reduce_all(
                red_fun,
                map_fun(Vec256::loadu(data as *const c_void, size)),
                size,
            )
        };
    }
    let mut d = vec_size;
    let mut acc_vec = map_fun(Vec256::loadu(data as *const c_void, None));
    let limit = size - (size % vec_size);
    while d < limit {
        let mut data_vec = Vec256::loadu(unsafe { data.offset(d as isize) as *const c_void }, None);
        data_vec = map_fun(data_vec);
        acc_vec = red_fun(acc_vec, data_vec);
        d += vec_size;
    }
    if size - d > 0 {
        let mut data_vec = Vec256::loadu(
            unsafe { data.offset(d as isize) as *const c_void },
            size - d,
        );
        data_vec = map_fun(data_vec);
        acc_vec = Vec256::set(&acc_vec, &red_fun(acc_vec.clone(), data_vec), size - d);
    }
    return unsafe { vec_reduce_all(red_fun, acc_vec, vec_size) };
}

#[inline(always)]
pub fn map2_reduce_all<T: num::Float, M, R>(
    mut map_fun: M,
    mut red_fun: R,
    data: *mut T,
    data2: *mut T,
    size: usize,
) -> T
where
    M: FnMut(Vec256<T>, Vec256<T>) -> Vec256<T>,
    R: FnMut(Vec256<T>, Vec256<T>) -> Vec256<T>,
{
    let vec_size = Vec256::<T>::size();
    if size < vec_size {
        return unsafe {
            let data_vec = Vec256::loadu(data as *const c_void, size);
            let data2_vec = Vec256::loadu(data2 as *const c_void, size);
            vec_reduce_all(red_fun, map_fun(data_vec, data2_vec), size)
        };
    }
    let mut d = vec_size;
    let mut acc_vec = map_fun(
        Vec256::loadu(data as *const c_void, None),
        Vec256::loadu(data2 as *const c_void, None),
    );
    let limit = size - (size % vec_size);
    while d < limit {
        let data_vec = Vec256::loadu(unsafe { data.add(d) as *const c_void }, None);
        let data2_vec = Vec256::loadu(unsafe { data2.add(d) as *const c_void }, None);
        acc_vec = red_fun(acc_vec, map_fun(data_vec, data2_vec));
        d += vec_size;
    }
    if size - d > 0 {
        let data_vec = Vec256::loadu(unsafe { data.add(d) as *const c_void }, size - d);
        let data2_vec = Vec256::loadu(unsafe { data2.add(d) as *const c_void }, size - d);
        acc_vec = Vec256::set(
            &acc_vec,
            &red_fun(acc_vec.clone(), map_fun(data_vec, data2_vec)),
            size - d,
        );
    }
    return unsafe { vec_reduce_all(red_fun, acc_vec, vec_size) };
}

#[inline(always)]
pub fn map<T: num::Float, M>(
    mut vec_fun: M,
    output_data: *mut T,
    input_data: *mut T,
    size: usize,
) -> ()
where
    M: FnMut(Vec256<T>) -> Vec256<T>,
{
    let mut d = 0;
    let vec_size = Vec256::<T>::size();
    let limit = size - (size % vec_size);
    while d < limit {
        let output_vec = vec_fun(Vec256::loadu(
            unsafe { input_data.add(d) as *const c_void },
            None,
        ));
        output_vec.store(unsafe { output_data.add(d) as *mut c_void }, None);
        d += vec_size;
    }

    if size - d > 0 {
        let output_vec = vec_fun(Vec256::loadu(
            unsafe { input_data.add(d) as *const c_void },
            size - d,
        ));
        output_vec.store(unsafe { output_data.add(d) as *mut c_void }, size - d);
    }
}

#[inline(always)]
pub fn map2<T: num::Float, M>(
    mut vec_fun: M,
    output_data: *mut T,
    input_data: *mut T,
    input_data2: *mut T,
    size: usize,
) -> ()
where
    M: FnMut(Vec256<T>, Vec256<T>) -> Vec256<T>,
{
    let mut d = 0;
    let vec_size = Vec256::<T>::size();
    let limit = size - (size % vec_size);
    while d < limit {
        let data_vec = Vec256::loadu(
            unsafe { input_data.offset(d as isize) as *const c_void },
            None,
        );
        let data_vec2 = Vec256::loadu(
            unsafe { input_data2.offset(d as isize) as *const c_void },
            None,
        );
        let output_vec = vec_fun(data_vec, data_vec2);
        output_vec.store(
            unsafe { output_data.offset(d as isize) as *mut c_void },
            None,
        );
        d += vec_size
    }

    if size - d > 0 {
        let data_vec = Vec256::loadu(
            unsafe { input_data.offset(d as isize) as *const c_void },
            size - d,
        );
        let data_vec2 = Vec256::loadu(
            unsafe { input_data2.offset(d as isize) as *const c_void },
            size - d,
        );
        let output_vec = vec_fun(data_vec, data_vec2);
        output_vec.store(
            unsafe { output_data.offset(d as isize) as *mut c_void },
            size - d,
        );
    }
}
