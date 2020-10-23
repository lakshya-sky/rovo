use crate::tensor::*;
use std::ptr::NonNull;

pub fn cpu_serial_kernel<F, T: num::cast::NumCast>(iter: TensorIterator, mut op: F)
where
    F: FnMut() -> T,
{
    let output = iter.output();
    let size = output.numel();
    let mut data_iter = output.get_tensor_impl().data.iter_mut();
    let mut i = 0;

    loop {
        if i >= size {
            break;
        }
        let elem = data_iter.next().unwrap();
        *elem = num::cast(op()).unwrap();
        i += 1;
    }
}

pub fn cpu_kernel_vec<F, T>(iter: &mut NewTensorIterator, op: F)
where
    F: FnMut() -> T + Copy,
{
    let closure = move |data: &[NonNull<u8>], _strides: &[usize], n: usize| {
        let op_ = op;
        vectorized_loop(data, n, 0, op_);
    };
    iter.for_each(closure);
}
pub fn vectorized_loop<F, T>(data_: &[NonNull<u8>], n: usize, s: usize, op: F)
where
    F: FnMut() -> T,
{
    let ntensors = data_.len();
    let mut data = Vec::<NonNull<u8>>::with_capacity(ntensors);
    for arg in 0..ntensors {
        data.push(data_[arg]);
    }

    let i = 0;
    if i < n {
        let mut strides = Vec::<usize>::with_capacity(ntensors);
        for arg in 0..ntensors {
            strides.push(if s > 0 && arg == s {
                0
            } else {
                std::mem::size_of::<T>()
            })
        }
        basic_loop(data.as_slice(), strides.as_slice(), i, n, op);
    }
}

pub fn basic_loop<F, T>(data: &[NonNull<u8>], strides_: &[usize], i: usize, n: usize, op: F)
where
    F: FnMut() -> T,
{
    let ntensors = data.len();
    let mut strides = Vec::<usize>::with_capacity(ntensors);
    for arg in 0..ntensors {
        strides.push(strides_[arg])
    }
    execute_op(data, strides.as_slice(), i, n, op);
}

pub fn execute_op<F, T>(data: &[NonNull<u8>], strides: &[usize], mut i: usize, n: usize, mut op: F)
where
    F: FnMut() -> T,
{
    loop {
        if i >= n {
            break;
        }
        unsafe {
            let out_ptr = data[0].as_ptr().add(i * strides[0]) as *mut T;
            *out_ptr = op();
        };
        i += 1;
    }
}
