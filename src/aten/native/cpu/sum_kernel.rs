use num::Float;

use crate::{
    aten::native::basic_loop, util::vec256::Vec256, Closure, AT_DISPATCH_FLOATING_TYPES_AND2,
};
use crate::{c10::isIntegralType, tensor::TensorIterator};
use std::{ffi::c_void, marker::PhantomData, mem::size_of, ptr::NonNull};

pub fn UNARY_OUTER_LOOP<F>(data: &mut [NonNull<u8>], strides: &[usize], n: usize, f: F) -> ()
where
    F: Fn(),
{
    for _ in 0..n {
        f();
        unsafe {
            data[0] = NonNull::new_unchecked(data[0].as_ptr().add(strides[0]));
            data[1] = NonNull::new_unchecked(data[1].as_ptr().add(strides[1]));
        }
    }
}

pub fn sum_kernel_impl(iter: &TensorIterator) -> () {
    if isIntegralType(iter.dtype(), true) {}
    AT_DISPATCH_FLOATING_TYPES_AND2!(_, _, iter.dtype(), "sum_cpu", move || {
        iter.output().fill_(0 as SCALART);
        iter.parallel_reduce(
            |data: &[NonNull<u8>], strides: &[usize], mut size0: usize, mut size1: usize| {
                let mut in_strides = [strides[1], strides[3]];
                let mut out_strides = [strides[0], strides[2]];
                if out_strides[0] != 0 && out_strides[1] == 0 {
                    in_strides.swap(0, 1);
                    out_strides.swap(0, 1);
                    std::mem::swap(&mut size0, &mut size1);
                }
                // Special case? - not a true reduction
                if out_strides[0] != 0 && out_strides[1] != 0 {
                    let outer_strides = [strides[2], strides[3]];
                    let mut data_param = data[0..2].to_vec();
                    UNARY_OUTER_LOOP(data_param.as_mut_slice(), &outer_strides, size1, || {
                        let ptrs = [data[0], data[0], data[1]];
                        let inner_strides = [strides[0], strides[0], strides[1]];
                        basic_loop(
                            &ptrs,
                            &inner_strides,
                            0,
                            size0,
                            &mut Closure::new(|args: [SCALART; 2]| {
                                return args[0] + args[1];
                            }),
                        );
                    });
                    return;
                }
                let out_stride = out_strides[1];
                assert_eq!(out_strides[0], 0);
                let vec256_size = Vec256::<SCALART>::size();
                if in_strides[0] == std::mem::size_of::<SCALART>() && size0 >= vec256_size {
                    // Contiguous inner reduction
                    //vectorized_inner_sum::<SCALART>(data, in_strides[1], out_stride, size0, size1);
                    todo!()
                } else if in_strides[1] == std::mem::size_of::<SCALART>() && size1 >= vec256_size {
                    // Contiguous outer reduction
                    vectorized_outer_sum::<SCALART>(data, in_strides[0], out_stride, size0, size1);
                } else if in_strides[0] < in_strides[1] {
                    todo!();
                    // scalar_inner_sum::<SCALART>(data, in_strides, out_stride, size0, size1);
                } else {
                    todo!();
                    // scalar_outer_sum::<SCALART>(data, in_strides, out_stride, size0, size1);
                }
            },
        );
    });
}
trait LoadImpl {
    type Output;
    fn load(data: *const c_void, stride: usize, index: usize) -> Self::Output;
}

impl<T: Float> LoadImpl for Vec256<T> {
    type Output = Vec256<T>;
    fn load(data: *const c_void, stride: usize, index: usize) -> Self::Output {
        let ptr = unsafe { data.add(index * stride) };
        Vec256::<T>::loadu(ptr, Self::Output::size())
    }
}

impl<T: Float> LoadImpl for T {
    type Output = T;
    fn load(data: *const c_void, stride: usize, index: usize) -> Self::Output {
        unsafe {
            let ptr = data.add(index * stride) as *const T;
            ptr.read()
        }
    }
}

fn load<T: LoadImpl>(data: *const u8, stride: usize, index: usize) -> T::Output {
    T::load(data as *const c_void, stride, index)
}
// struct LoadImpl<T> {
//     _ph: PhantomData<T>,
// }

// impl<T: Float> LoadImpl<Vec256<T>> {
//     fn load(data: *const c_void, stride: usize, index: usize) -> Vec256<T> {
//         let ptr = unsafe { data.add(index * stride) };
//         Vec256::<T>::loadu(ptr, Vec256::<T>::size())
//     }
// }

// impl<T: Float> LoadImpl<T> {
//     fn load(data: *const c_void, stride: usize, index: usize) -> T {
//         unsafe {
//             let ptr = data.add(index * stride) as *const T;
//             ptr.read()
//         }
//     }
// }

// fn load<T: Float>(data: *const u8, stride: usize, index: usize) -> T {
//     LoadImpl::<T>::load(data as *const c_void, stride, index)
// }

// fn load<V>(data: *const u8, stride: usize, index: usize) -> T {
//     LoadImpl::<T>::load(data as *const c_void, stride, index)
// }
#[inline(always)]
fn ceil_log2(x: usize) -> usize {
    if x <= 2 {
        1
    } else {
        ((x as f64).log2().ceil()) as usize
    }
}

struct MultiRowSum<T>(PhantomData<T>);
impl<T: Float> MultiRowSum<T> {
    fn call(
        in_data: *const u8,
        row_stride: usize,
        col_stride: usize,
        size: usize,
        nrows: usize,
    ) -> Vec<T> {
        let num_levels = 4;
        //Todo: This operation of finding ceil_log2 may result in incorrect result;
        let level_power = 4usize.max(ceil_log2(size) / num_levels);
        let level_step = 1 << level_power;
        let level_mask = level_step - 1;
        let mut acc = vec![vec![T::zero(); nrows]; num_levels];
        let mut i = 0;
        while i + level_step <= size {
            for j in 0..level_step {
                let sum_base = unsafe { in_data.add(i * row_stride) };
                for k in 0..nrows {
                    acc[0][k] =
                        acc[0][k].clone() + T::load(sum_base as *const c_void, col_stride, k);
                }
                i += 1;
            }

            for j in 1..num_levels {
                for k in 0..nrows {
                    acc[j][k] = acc[j][k] + acc[j - 1][k];
                    acc[j - 1][k] = T::zero();
                }
                let mask = level_mask << (j * level_power);
                if (i & mask) != 0 {
                    break;
                }
            }
        }
        while i < size {
            let sum_base = unsafe { in_data.add(i * row_stride) as *const c_void };
            for k in 0..nrows {
                acc[0][k] = acc[0][k] + T::load(sum_base, col_stride, k);
            }
            i += 1;
        }
        for j in 1..num_levels {
            for k in 0..nrows {
                acc[0][k] = acc[0][k] + acc[j][k];
            }
        }

        let mut ret = Vec::with_capacity(nrows);
        for k in 0..nrows {
            ret.push(acc[0][k].clone());
        }
        return ret;
    }
}

impl<T: Float> MultiRowSum<Vec256<T>> {
    fn call(
        in_data: *const u8,
        row_stride: usize,
        col_stride: usize,
        size: usize,
        nrows: usize,
    ) -> Vec<Vec256<T>> {
        let num_levels = 4;
        //Todo: This operation of finding ceil_log2 may result in incorrect result;
        let level_power = 4usize.max(ceil_log2(size) / num_levels);
        let level_step = 1 << level_power;
        let level_mask = level_step - 1;
        let mut acc = vec![vec![Vec256::<T>::new(); nrows]; num_levels];
        let mut i = 0;
        while i + level_step <= size {
            for j in 0..level_step {
                let sum_base = unsafe { in_data.add(i * row_stride) };
                for k in 0..nrows {
                    acc[0][k] = acc[0][k].clone()
                        + Vec256::<T>::load(sum_base as *const c_void, col_stride, k);
                }
                i += 1;
            }

            for j in 1..num_levels {
                for k in 0..nrows {
                    acc[j][k] = &acc[j][k] + &acc[j - 1][k];
                    acc[j - 1][k] = Vec256::<T>::filled_new(T::zero());
                }
                let mask = level_mask << (j * level_power);
                if (i & mask) != 0 {
                    break;
                }
            }
        }
        while i < size {
            let sum_base = unsafe { in_data.add(i * row_stride) as *const c_void };
            for k in 0..nrows {
                acc[0][k] = &acc[0][k] + &Vec256::<T>::load(sum_base, col_stride, k);
            }
            i += 1;
        }
        for j in 1..num_levels {
            for k in 0..nrows {
                acc[0][k] = &acc[0][k] + &acc[j][k];
            }
        }

        let mut ret = Vec::with_capacity(nrows);
        for k in 0..nrows {
            ret.push(acc[0][k].clone());
        }
        return ret;
    }
}
struct RowSum<T>(PhantomData<T>);
impl<T: Float> RowSum<T> {
    pub fn call(in_data: *const u8, in_stride: usize, size: usize) -> T {
        let ilp_factor = 4;
        // Interpret row as a (-1, ilp_factor) shaped array to find partial sums
        let size_ilp = size / ilp_factor;
        let mut partial_sums = MultiRowSum::<T>::call(
            in_data,
            in_stride * ilp_factor,
            in_stride,
            size_ilp,
            ilp_factor,
        );
        for i in size_ilp * ilp_factor..size {
            partial_sums[0] = partial_sums[0] + T::load(in_data as *const c_void, in_stride, i);
        }
        for k in 1..ilp_factor {
            partial_sums[0] = partial_sums[0] + partial_sums[k];
        }
        return partial_sums.remove(0);
    }
}
impl<T: Float> RowSum<Vec256<T>> {
    pub fn call(in_data: *const u8, in_stride: usize, size: usize) -> Vec256<T> {
        let ilp_factor = 4;
        // Interpret row as a (-1, ilp_factor) shaped array to find partial sums
        let size_ilp = size / ilp_factor;
        let mut partial_sums = MultiRowSum::<Vec256<T>>::call(
            in_data,
            in_stride * ilp_factor,
            in_stride,
            size_ilp,
            ilp_factor,
        );
        for i in size_ilp * ilp_factor..size {
            partial_sums[0] =
                &partial_sums[0] + &Vec256::<T>::load(in_data as *const c_void, in_stride, i);
        }
        for k in 1..ilp_factor {
            partial_sums[0] = &partial_sums[0] + &partial_sums[k];
        }
        return partial_sums.remove(0);
    }
}

pub fn accumulate_result<T: Float>(data: *mut u8, stride: usize, index: usize, value: T) {
    unsafe {
        let ptr = data.add(index * stride) as *mut T;
        ptr.write(ptr.read() + value)
    }
}

fn accumulate_result_array<T: Float>(data: *mut u8, stride: usize, index: usize, values: &[T]) {
    let base_ptr = unsafe { data.add(stride * index) };
    for (k, v) in values.iter().enumerate() {
        accumulate_result(base_ptr, stride, k, *v)
    }
}

fn vectorized_outer_sum<T: Float>(
    data: &[NonNull<u8>],
    inner_stride: usize,
    out_stride: usize,
    size0: usize,
    size1: usize,
) {
    let nrows = 4;
    let vec_size = Vec256::<T>::size();
    let type_size = size_of::<T>();
    let vec_stride = vec_size * type_size;
    // Input is contiguous over the second (non-reduced) dimension
    let mut j = 0;
    loop {
        if j + nrows * vec_size > size1 {
            break;
        }
        let row_in = unsafe { data[1].as_ptr().add(j * type_size) };
        let sums = MultiRowSum::<Vec256<T>>::call(row_in, inner_stride, vec_stride, size0, nrows);
        for i in 0..nrows {
            let base_idx = j + i * vec_size;
            let mut ans = vec![T::zero(); Vec256::<T>::size()];
            sums[i].store(ans.as_mut_ptr() as *mut c_void, None);
            accumulate_result_array(data[0].as_ptr(), out_stride, base_idx, ans.as_slice());
        }
        j += nrows * vec_size;
    }
    loop {
        if j + vec_size > size1 {
            break;
        }
        let row_in = unsafe { data[1].as_ptr().add(j * size_of::<T>()) };
        let sums = RowSum::<Vec256<T>>::call(row_in, inner_stride, size0);
        let mut ans = vec![T::zero(); Vec256::<T>::size()];
        sums.store(ans.as_mut_ptr() as *mut c_void, None);
        accumulate_result_array(data[0].as_ptr(), out_stride, j, ans.as_slice());
        j += vec_size;
    }

    loop {
        if j >= size1 {
            break;
        }
        let row_in = unsafe { data[1].as_ptr().add(j * size_of::<T>()) };
        let ans = RowSum::<T>::call(row_in, inner_stride, size0);
        accumulate_result(data[0].as_ptr(), out_stride, j, ans);
        j += 1;
    }
}
