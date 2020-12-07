use crate::tensor::*;
use crate::Closure;
use std::convert::TryInto;
use std::ptr::NonNull;

// struct IsContiguous<N,SI,T,S>;

pub fn cpu_serial_kernel<I: Copy, O: num::cast::NumCast, F, const N: usize>(
    iter: &mut TensorIterator,
    mut op: Closure<I, O, F, N>,
) where
    F: FnMut([I; N]) -> O,
{
    let closure = move |data: &[NonNull<u8>], strides: &[usize], n: usize| {
        basic_loop(data, strides, 0, n, &mut op);
    };

    let range = 0..iter.numel();
    iter.serial_for_each(closure, range);
}

fn eval(
    strides: &[usize],
    n: usize,
    stride_index: isize,
    s: isize,
    input_type_size: usize,
    output_type_size: usize,
) -> bool {
    if n == 0 && stride_index == 0 {
        strides[0] == output_type_size
    } else if n == 0 && stride_index == -1 {
        true
    } else {
        let first_bool =
            strides[stride_index as usize] == if s == n as isize { 0 } else { input_type_size };
        first_bool
            && eval(
                strides,
                n - 1,
                stride_index - 1,
                s,
                input_type_size,
                output_type_size,
            )
    }
}

fn is_contiguous(
    strides: &[usize],
    arity: usize,
    input_type_size: usize,
    output_type_size: usize,
) -> bool {
    let n: usize = arity;

    let stride_index = if output_type_size == 0 {
        (arity - 1) as isize
    } else {
        arity as isize
    };

    let s = -1;

    eval(
        strides,
        n,
        stride_index,
        s,
        input_type_size,
        output_type_size,
    )
}

fn is_contiguous_scalar(
    strides: &[usize],
    s: usize,
    arity: usize,
    input_type_size: usize,
    output_type_size: usize,
) -> bool {
    let n: usize = arity;
    let stride_index = if output_type_size == 0 {
        (arity - 1) as isize
    } else {
        arity as isize
    };
    eval(
        strides,
        n,
        stride_index,
        s as isize,
        input_type_size,
        output_type_size,
    )
}

fn unroll_contiguous_scalar_checks(
    strides: &[usize],
    mut indices: std::ops::Range<usize>,
    arity: usize,
    input_type_size: usize,
    output_type_size: usize,
    mut cb: impl FnMut(usize),
) {
    if let Some(range_start) = indices.next() {
        if is_contiguous_scalar(
            strides,
            range_start,
            arity,
            input_type_size,
            output_type_size,
        ) {
            cb(range_start + 1);
        } else {
            unroll_contiguous_scalar_checks(
                strides,
                indices,
                arity,
                input_type_size,
                output_type_size,
                cb,
            )
        }
    } else {
        cb(0)
    }
}

pub fn cpu_kernel_vec<I: Copy, O: num::cast::NumCast, F, const N: usize>(
    iter: &mut TensorIterator,
    mut op: Closure<I, O, F, N>,
) where
    F: FnMut([I; N]) -> O,
{
    let closure = move |data: &[NonNull<u8>], strides: &[usize], n: usize| {
        if is_contiguous(
            strides,
            op.arity(),
            op.input_type_size(),
            op.output_type_size(),
        ) {
            vectorized_loop(data, n, 0, &mut op);
        } else {
            let arity = op.arity();
            let indices = 0..arity;
            let (in_size, out_size) = (op.input_type_size(), op.output_type_size());
            unroll_contiguous_scalar_checks(
                strides,
                indices,
                arity,
                in_size,
                out_size,
                |idx: usize| {
                    if idx != 0 {
                        vectorized_loop(data, n, idx, &mut op)
                    } else {
                        basic_loop(data, strides, 0, n, &mut op)
                    }
                },
            );
        }
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
