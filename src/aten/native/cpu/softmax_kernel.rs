use std::sync::atomic::AtomicPtr;

use crate::{
    aten::{parallel_for, GRAIN_SIZE},
    c10::ScalarType,
    tensor::Tensor,
    util::vec256::{self, Vec256},
    AT_DISPATCH_FLOATING_TYPES_AND2, AT_PRIVATE_CASE_TYPE,
};

#[inline(always)]
fn _vec_log_softmax_lastdim<T: num::Float + PartialOrd, const CHUNK_SIZE: usize>(
    input_data_base: *mut T,
    output_data_base: *mut T,
    outer_size: usize,
    dim_size: usize,
) {
    // Note: chunk_size is used to store 4k bytes of the T on the stack.
    // Pytorch uses vec256 here to store 256bits or(32 bytes of T).
    // I couldn't create chunk_size as const here so passed as const generic.
    let mut grain_size = GRAIN_SIZE / (16 * dim_size * CHUNK_SIZE);
    if grain_size < CHUNK_SIZE {
        grain_size = CHUNK_SIZE;
    }
    let mut input_data_base = AtomicPtr::new(input_data_base);
    let mut output_data_base = AtomicPtr::new(output_data_base);

    let closure = |begin: usize, end: usize| {
        for ii in (begin..end).step_by(CHUNK_SIZE) {
            let mut tmp_sum_scalar = [T::zero(); CHUNK_SIZE];
            let mut max_input_arr = [T::zero(); CHUNK_SIZE];
            let mut loop_end = CHUNK_SIZE;
            if ii + CHUNK_SIZE > end {
                loop_end = end - ii;
            }

            for j in 0..loop_end {
                let i = ii + j;
                let input_data =
                    unsafe { input_data_base.get_mut().offset((i * dim_size) as isize) };
                max_input_arr[j] = vec256::reduce_all(
                    |x: Vec256<T>, y: Vec256<T>| -> Vec256<T> {
                        return Vec256::maximum(&x, &y);
                    },
                    input_data,
                    dim_size,
                );
            }
            for j in 0..loop_end {
                let i = ii + j;
                let input_data =
                    unsafe { input_data_base.get_mut().offset((i * dim_size) as isize) };
                let max_input = max_input_arr[j];
                tmp_sum_scalar[j] = vec256::map_reduce_all(
                    move |x: Vec256<T>| {
                        return (x - Vec256::filled_new(max_input)).exp();
                    },
                    |x: Vec256<T>, y: Vec256<T>| x + y,
                    input_data,
                    dim_size,
                );
            }
            // See [Note AVX-SSE transitions] for why this should call the
            // vectorized version (aside from perf improvements).
            vec256::map(
                |x: Vec256<T>| {
                    return x.log();
                },
                tmp_sum_scalar.as_mut_ptr(),
                tmp_sum_scalar.as_mut_ptr(),
                loop_end,
            );
            for j in 0..loop_end {
                let i = ii + j;
                let input_data =
                    unsafe { input_data_base.get_mut().offset((i * dim_size) as isize) };
                let output_data =
                    unsafe { output_data_base.get_mut().offset((i * dim_size) as isize) };

                let tmp_sum = tmp_sum_scalar[j];
                let max_input = max_input_arr[j];

                // It's necessary to keep the order of the operations below.
                // In some cases that input is large digits and the difference
                // is small, if we compute `max_input` plus `tmp_sum` before,
                // there would be a numerical problem. See an example in
                // https://github.com/pytorch/pytorch/issues/11752#issuecomment-422883379
                vec256::map(
                    |x: Vec256<T>| {
                        return x - Vec256::filled_new(max_input) - Vec256::filled_new(tmp_sum);
                    },
                    output_data,
                    input_data,
                    dim_size,
                );
            }
        }
    };
    parallel_for(0, outer_size, grain_size, closure);
}
struct vec_host_softmax_lastdim;
impl vec_host_softmax_lastdim {
    #[inline(always)]
    pub fn apply<SCALART: num::Float + PartialOrd, const CHUNK_SIZE: usize>(
        output: &Tensor,
        input: &Tensor,
        log_softmax: bool,
    ) {
        let mut outer_size = 1;
        let dim_size = input.size((input.ndimension() - 1) as isize);
        for i in 0..input.ndimension() - 1 {
            outer_size *= input.size(i as isize);
        }

        let input_data_base = input.data_ptr_casted::<SCALART>();
        let output_data_base = output.data_ptr_casted::<SCALART>();
        if log_softmax {
            _vec_log_softmax_lastdim::<SCALART, CHUNK_SIZE>(
                input_data_base,
                output_data_base,
                outer_size,
                dim_size,
            );
        } else {
            // _vec_softmax_lastdim(input_data_base, output_data_base, outer_size, dim_size);
        }
    }
}

#[inline(always)]
pub fn log_softmax_lastdim_kernel_impl(result: &Tensor, self_: &Tensor) {
    AT_DISPATCH_FLOATING_TYPES_AND2!(
        self_.scalar_type(),
        "log_softmax_lastdim_kernel_impl",
        || {
            const scalar_size: usize = std::mem::size_of::<SCALART>();
            const CHUNK_SIZE: usize = (128 / scalar_size) * (32 * scalar_size);
            vec_host_softmax_lastdim::apply::<SCALART, CHUNK_SIZE>(result, self_, true);
        }
    );
}
