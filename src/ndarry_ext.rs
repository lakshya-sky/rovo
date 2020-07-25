use ndarray;
use num;
use num_traits;
use std::fmt;


pub type NdArray<T> = ndarray::Array<T, ndarray::IxDyn>;
pub type NdArrayView<'a, T> = ndarray::ArrayView<'a, T, ndarray::IxDyn>;
pub trait Float:
    num_traits::Float
    + num_traits::NumAssignOps
    + Copy
    + Send
    + Sync
    + fmt::Display
    + fmt::Debug
    + Sized
    + 'static
{
}

impl<T> Float for T where
    T: num::Float
        + num_traits::NumAssignOps
        + Copy
        + Send
        + Sync
        + fmt::Display
        + fmt::Debug
        + Sized
        + 'static
{
}



#[inline]
/// Creates an ndarray filled with 0s.
pub fn zeros<T: Float>(shape: &[usize]) -> NdArray<T> {
    NdArray::from_elem(shape, T::zero())
}
#[inline]
/// Creates an ndarray filled with 1s.
pub fn ones<T: Float>(shape: &[usize]) -> NdArray<T> {
    NdArray::from_elem(shape, T::one())
}

#[inline]
/// Creates an ndarray filled with 1s.
pub fn from_elem<T: Float>(shape: &[usize], n: T) -> NdArray<T> {
    NdArray::from_elem(shape, T::from(n).unwrap())
}

#[inline]
/// Creates an ndarray object from a scalar.
pub fn from_scalar<T: Float>(val: T) -> NdArray<T> {
    NdArray::from_elem(ndarray::IxDyn(&[]), val)
}

#[inline]
pub fn add<T: Float>(input_1: NdArray<T>, input_2: NdArray<T>) -> NdArray<T> {
    input_1 + input_2
}

#[cfg(test)]
mod tests {
    use crate::ndarry_ext::*;
    use ndarray::prelude::*;
    #[test]
    fn test_init_zeros() {
        let array = zeros::<f64>(&[2, 2]);
        assert_eq!(vec![2, 2], array.shape());
        assert_eq!(vec![0.0, 0.0, 0.0, 0.0], array.into_raw_vec());
    }
    #[test]
    fn test_init_ones() {
        let array = ones::<f64>(&[2, 2]);
        assert_eq!(vec![2, 2], array.shape());
        assert_eq!(vec![1.0, 1.0, 1.0, 1.0], array.into_raw_vec());
    }
    #[test]
    fn test_init_from_elem() {
        let array = from_elem::<f64>(&[2, 1], 3.0);
        assert_eq!(vec![2, 1], array.shape());
        assert_eq!(vec![3.0, 3.0], array.into_raw_vec());
    }
    #[test]
    fn test_init_scalar_sum() {
        let array_1 = from_scalar::<f64>(2.0);
        let array_2 = from_scalar::<f64>(1.0);
        assert_eq!(vec![3.0], add(array_1, array_2).into_raw_vec())
    }
    #[test]
    fn test_array_sum() {
        let array_1 = from_elem::<f64>(&[2, 1], 2.0);
        let array_2 = from_elem::<f64>(&[2, 1], 3.0);
        // print!("{:?}", add(array_1, array_2));
        assert_eq!(
            array![[5.0], [5.0]],
            add(array_1, array_2).into_shape((2, 1)).unwrap()
        );
    }
}
