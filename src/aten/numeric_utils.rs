use num::{Bounded, Float};

pub trait IsNaN {
    fn is_nan(&self) -> bool;
}

macro_rules! impl_IsNaN_int {
    ($rt: expr, $($t:ty),+) => {
        $(impl IsNaN for $t {
            fn is_nan(&self) -> bool {
                $rt
            }
        })*
    }
}

macro_rules! impl_IsNaN_float{
    ($($t:ty),+) => {
        $(impl IsNaN for $t{
            fn is_nan(&self)->bool{
                Float::is_nan(*self)
            }
        })*
    }
}

impl_IsNaN_int!(false, u8, u16, u32, u64, i8, i16, i32, i64);
impl_IsNaN_float!(f32, f64);

pub fn is_nan<T: IsNaN>(n: T) -> bool {
    n.is_nan()
}

#[inline(always)]
pub fn lower_bound<T: Bounded>() -> T {
    return T::min_value();
}

#[inline(always)]
pub fn upper_bound<T: Bounded>() -> T {
    return T::max_value();
}
