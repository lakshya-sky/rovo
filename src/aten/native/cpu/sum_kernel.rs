use std::ptr::NonNull;

use crate::tensor::TensorIterator;

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

pub fn sum_kernel_impl(_iter: &TensorIterator) -> () {}

/*
pub fn sum_kernel_impl(iter: &TensorIterator) -> () {
    if isIntegralType(iter.dtype(), true) {}
    AT_DISPATCH_FLOATING_TYPES_AND2!(iter.dtype(), "sum_cpu", || {
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
                        aten::native::basic_loop(
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
                const vec_N: usize = 32 / std::mem::size_of::<SCALART>();

                // if in_strides[0] == std::mem::size_of::<SCALART>()
                //     && size0 >= Vec256::<SCALART, vec_N>::size()
                // {
                //     // Contiguous inner reduction
                //     vectorized_inner_sum::<SCALART>(data, in_strides[1], out_stride, size0, size1);
                // } else if in_strides[1] == std::mem::size_of::<SCALART>()
                //     && size1 >= Vec256::<SCALART, vec_N>::size()
                // {
                //     // Contiguous outer reduction
                //     vectorized_outer_sum::<SCALART>(data, in_strides[0], out_stride, size0, size1);
                // } else if in_strides[0] < in_strides[1] {
                //     scalar_inner_sum::<SCALART>(data, in_strides, out_stride, size0, size1);
                // } else {
                //     scalar_outer_sum::<SCALART>(data, in_strides, out_stride, size0, size1);
                // }
            },
        );
    });
}
*/

/*
template <typename scalar_t, int64_t nrows>
std::array<scalar_t, nrows> multi_row_sum(
    const char * C10_RESTRICT in_data,
    const int64_t row_stride,
    const int64_t col_stride,
    const int64_t size) {
  constexpr int64_t num_levels = 4;

  const int64_t level_power =
      std::max(int64_t(4), ceil_log2(size) / num_levels);
  const int64_t level_step = (1 << level_power);
  const int64_t level_mask = level_step - 1;

  scalar_t acc[num_levels][nrows];
  std::fill_n(&acc[0][0], num_levels * nrows, scalar_t(0));

  int64_t i = 0;
  for (; i + level_step <= size;) {
    for (int64_t j = 0; j < level_step; ++j, ++i) {
      const char * sum_base = in_data + i * row_stride;
      #pragma unroll
      for (int64_t k = 0; k < nrows; ++k) {
        acc[0][k] += load<scalar_t>(sum_base, col_stride, k);
      }
    }

    for (int64_t j = 1; j < num_levels; ++j) {
      #pragma unroll
      for (int64_t k = 0; k < nrows; ++k) {
        acc[j][k] += acc[j-1][k];
        acc[j-1][k] = scalar_t(0);
      }

      const auto mask = (level_mask << (j * level_power));
      if ((i & mask) != 0) {
        break;
      }
    }
  }

  for (; i < size; ++i) {
    const char * sum_base = in_data + i * row_stride;
    #pragma unroll
    for (int64_t k = 0; k < nrows; ++k) {
      acc[0][k] += load<scalar_t>(sum_base, col_stride, k);
    }
  }

  for (int64_t j = 1; j < num_levels; ++j) {
    #pragma unroll
    for (int64_t k = 0; k < nrows; ++k) {
      acc[0][k] += acc[j][k];
    }
  }

  std::array<scalar_t, nrows> ret;
  for (int64_t k = 0; k < nrows; ++k) {
    ret[k] = acc[0][k];
  }
  return ret;
}

template <typename scalar_t>
scalar_t row_sum(const char * C10_RESTRICT in_data,
                 const int64_t in_stride, const int64_t size) {
  constexpr int64_t ilp_factor = 4;

  // Interpret row as a (-1, ilp_factor) shaped array to find partial sums
  const int64_t size_ilp = size / ilp_factor;
  auto partial_sums = multi_row_sum<scalar_t, ilp_factor>(
      in_data, in_stride * ilp_factor, in_stride, size_ilp);

  for (int64_t i = size_ilp * ilp_factor; i < size; ++i) {
    partial_sums[0] += load<scalar_t>(in_data, in_stride, i);
  }

  for (int64_t k = 1; k < ilp_factor; ++k) {
    partial_sums[0] += partial_sums[k];
  }

  return partial_sums[0];
}

fn get_size<T>()->usize{
  return std::mem::size_of::<T>();
}

fn vectorized_inner_sum<T>(
    data: &[NonNull<u8>],
    outer_stride: usize,
    out_stride: usize,
    size0: usize,
    size1: usize,
) -> () {
    const N: usize = get_size::<T>();
    let vec_stride: usize = Vec256::<T, N>::size() * size_of::<T>();
    let vec_size: usize = size0 / Vec256::<T, N>::size();

    // // Input is contiguous over the first (reduced) dimension
    // for (int64_t j = 0; j < size1; ++j) {
    //   const auto *row_in = data[1] + j * outer_stride;
    //   auto vec_acc = row_sum<vec_t>(row_in, vec_stride, vec_size);

    //   scalar_t final_acc = 0;
    //   for (int64_t k = vec_size * vec_t::size(); k < size0; ++k) {
    //     final_acc += load<scalar_t>(row_in, sizeof(scalar_t), k);
    //   }

    //   scalar_t partials[vec_t::size()];
    //   vec_acc.store(partials);
    //   for (int64_t k = 0; k < vec_t::size(); ++k) {
    //     final_acc += partials[k];
    //   }
    //   accumulate_result(data[0], out_stride, j, final_acc);
    // }
}


template <typename scalar_t>
void scalar_inner_sum(
    char * C10_RESTRICT data[2], int64_t in_strides[2], int64_t out_stride,
    int64_t size0, int64_t size1) {
  for (int64_t j = 0; j < size1; ++j) {
    const auto *row_in = data[1] + j * in_strides[1];
    scalar_t ans = row_sum<scalar_t>(row_in, in_strides[0], size0);
    accumulate_result(data[0], out_stride, j, ans);
  }
}

template <typename scalar_t>
void vectorized_outer_sum(
    char * C10_RESTRICT data[2], int64_t inner_stride, int64_t out_stride,
    int64_t size0, int64_t size1) {
  using vec_t = Vec256<scalar_t>;
  constexpr int64_t nrows = 4;
  constexpr int64_t vec_stride = vec_t::size() * sizeof(scalar_t);

  // Input is contiguous over the second (non-reduced) dimension
  int64_t j = 0;
  for (; j + nrows * vec_t::size() <= size1; j += nrows * vec_t::size()) {
    const auto *row_in = data[1] + j * sizeof(scalar_t);
    auto sums = multi_row_sum<vec_t, nrows>(row_in, inner_stride, vec_stride, size0);

    for (int64_t i = 0; i < nrows; ++i) {
      const int64_t base_idx = j + i * vec_t::size();

      std::array<scalar_t, vec_t::size()> ans;
      sums[i].store(ans.data());
      accumulate_result(data[0], out_stride, base_idx, ans);
    }
  }

  for (; j + vec_t::size() <= size1; j += vec_t::size()) {
    const auto *row_in = data[1] + j * sizeof(scalar_t);
    const vec_t sums = row_sum<vec_t>(row_in, inner_stride, size0);

    std::array<scalar_t, vec_t::size()> ans;
    sums.store(ans.data());
    accumulate_result(data[0], out_stride, j, ans);
  }

  for (; j < size1; ++j) {
    const auto *row_in = data[1] + j * sizeof(scalar_t);
    scalar_t ans = row_sum<scalar_t>(row_in, inner_stride, size0);
    accumulate_result(data[0], out_stride, j, ans);
  }
}

template <typename scalar_t>
void scalar_outer_sum(
    char * C10_RESTRICT data[2], int64_t in_strides[2], int64_t out_stride,
    int64_t size0, int64_t size1) {

  constexpr int64_t nrows = 4;
  int64_t j = 0;
  for (; j + (nrows - 1) < size1; j += nrows) {
    const auto *row_in = data[1] + j * in_strides[1];
    auto sums = multi_row_sum<scalar_t, nrows>(
        row_in, in_strides[0], in_strides[1], size0);
    accumulate_result(data[0], out_stride, j, sums);
  }

  for (; j < size1; ++j) {
    const auto *row_in = data[1] + j * in_strides[1];
    scalar_t ans = row_sum<scalar_t>(row_in, in_strides[0], size0);
    accumulate_result(data[0], out_stride, j, ans);
  }
    }

*/
