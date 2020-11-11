use crate::tensor::*;
use crate::Closure;
use std::convert::TryInto;
use std::ptr::NonNull;

pub fn cpu_serial_kernel<I: Copy, O: num::cast::NumCast, F, const N: usize>(
    iter: &mut NewTensorIterator,
    mut op: Closure<I, O, F, N>,
) where
    F: FnMut([I; N]) -> O,
{
    let closure = move |data: &[NonNull<u8>], strides: &[usize], n: usize| {
        basic_loop(data, strides, 0, n, &mut op);
    };

    let range: Vec<usize> = (0..iter.numel()).collect();
    iter.serial_for_each(closure, range.as_slice());
}

pub fn cpu_kernel_vec<I: Copy, O: num::cast::NumCast, F, const N: usize>(
    iter: &mut NewTensorIterator,
    mut op: Closure<I, O, F, N>,
) where
    F: FnMut([I; N]) -> O,
{
    let closure = move |data: &[NonNull<u8>], _strides: &[usize], n: usize| {
        vectorized_loop(data, n, 0, &mut op);
    };
    iter.for_each(closure);
}
pub fn vectorized_loop<I: Copy, O: num::cast::NumCast, F, const N: usize>(
    data_: &[NonNull<u8>],
    n: usize,
    s: usize,
    op: &mut Closure<I, O, F, N>,
) where
    F: FnMut([I; N]) -> O,
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
                std::mem::size_of::<O>()
            })
        }
        basic_loop(data.as_slice(), strides.as_slice(), i, n, op);
    }
}

pub fn basic_loop<I: Copy, O: num::cast::NumCast, F, const N: usize>(
    data: &[NonNull<u8>],
    strides_: &[usize],
    i: usize,
    n: usize,
    op: &mut Closure<I, O, F, N>,
) where
    F: FnMut([I; N]) -> O,
{
    let ntensors = data.len();
    let mut strides = Vec::<usize>::with_capacity(ntensors);
    for arg in 0..ntensors {
        strides.push(strides_[arg])
    }
    execute_op(data, strides.as_slice(), i, n, op);
}

pub fn execute_op<I: Copy, O: num::cast::NumCast, F, const N: usize>(
    data: &[NonNull<u8>],
    strides: &[usize],
    mut i: usize,
    n: usize,
    op: &mut Closure<I, O, F, N>,
) where
    F: FnMut([I; N]) -> O,
{
    loop {
        if i >= n {
            break;
        }
        // Todo: Handle case where iterator has no output. here it is assumed that first element is output pointer.
        unsafe {
            let out_ptr = data[0].as_ptr().add(i * strides[0]) as *mut O;
            let inputs = dereference::<I, N>(
                data.split_first().unwrap().1,
                strides.split_first().unwrap().1,
                i,
            );
            *out_ptr = op.call(inputs);
        };
        i += 1;
    }
}
fn dereference<'a, I: Copy, const N: usize>(
    data: &'a [NonNull<u8>],
    strides: &[usize],
    i: usize,
) -> [I; N] {
    let base = &data[0..N];
    let stride = &strides[0..N];
    let mut inputs = Vec::with_capacity(N);
    for (b, s) in base.iter().zip(stride.iter()) {
        unsafe {
            let ptr_ = b.as_ptr().add(i * s) as *const I;
            inputs.push(*ptr_);
        }
    }
    let boxed_slice = inputs.into_boxed_slice();
    let boxed_array: Box<[I; N]> = match boxed_slice.try_into() {
        Ok(ba) => ba,
        Err(o) => panic!("Expected a Vec of length {} but it was {}", N, o.len()),
    };
    *boxed_array
}
