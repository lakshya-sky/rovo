use crate::ndarry_ext::*;

macro_rules! impl_bin_op_forward {
    ($forward_name:ident, $bin_op:tt, $vms_op:ident, $vmd_op:ident) => {
        pub fn $forward_name<'v, T: Float>(x0: &NdArrayView<'v, T>, x1: &NdArrayView<'v, T>) -> NdArray<T>
        {
            let shape0: &[usize] = x0.shape();
            let shape1: &[usize] = x1.shape();
            let scalar_shape = &[];
            let scalar_shape1 = &[0];

            let x0_is_scalar = shape0 == scalar_shape || shape0 == scalar_shape1;
            let x1_is_scalar = shape1 == scalar_shape || shape1 == scalar_shape1;

            if x0_is_scalar && !x1_is_scalar {
                let elem = x0[ndarray::IxDyn(&[])];
                x1.map(move |&a| a $bin_op elem)
            } else if x1_is_scalar && !x0_is_scalar {
                let elem = x1[ndarray::IxDyn(&[])];
                x0.map(move |&a| a $bin_op elem )
            } else if !x0_is_scalar && !x1_is_scalar {
                let len0: usize = shape0.iter().product();
                let len1: usize = shape1.iter().product();
                if len0 > len1 {
                    x0 $bin_op x1
                } else {
                    // tensor vs tensor (same shapes)
                    #[cfg(feature = "mkl")]
                    {
                        use crate::{ops::mkl_ffi::*, same_type};
                        bin_op_same_shape!($vms_op, $vmd_op, $bin_op, x0, x1)
                    }
                    #[cfg(not(feature = "mkl"))] {
                        x0 $bin_op x1
                    }
                }
            } else {
                // scalar vs scalar
                x0 $bin_op x1
            }
        }
    };
}

impl_bin_op_forward!(add_forward, +, vsAdd, vdAdd);
impl_bin_op_forward!(mul_forward, *, vsMul, vdMul);
