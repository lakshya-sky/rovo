use crate::c10::checked_convert;
use num::{cast, FromPrimitive, NumCast, ToPrimitive};
use std::any::type_name;
#[repr(u32)]
#[derive(Copy, Clone)]
enum Tag {
    I,
    F,
}

#[repr(C)]
#[derive(Copy, Clone)]
enum V {
    i(i64),
    f(f64),
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Scalar {
    v: V,
}

impl Default for Scalar {
    fn default() -> Self {
        Self::int(0)
    }
}

impl Scalar {
    #[inline(always)]
    pub fn int<T>(v: T) -> Self
    where
        T: NumCast,
    {
        let v = V::i(cast(v).unwrap());
        Self { v }
    }
    #[inline(always)]
    pub fn float<T>(v: T) -> Self
    where
        T: NumCast,
    {
        let v = V::f(cast(v).unwrap());
        Self { v }
    }
    #[inline(always)]
    pub fn new<T>(v: T) -> Self
    where
        T: ToPrimitive + NumCast,
    {
        let is_float = match type_name::<T>() {
            "f32" | "f64" => true,
            _ => false,
        };
        if is_float {
            Self::float(v)
        } else {
            Self::int(v)
        }
    }
    #[inline(always)]
    pub fn to<T: FromPrimitive>(&self) -> T {
        match type_name::<T>() {
            "f32" => FromPrimitive::from_f32(self.to_float()).unwrap(),
            "f64" => FromPrimitive::from_f64(self.to_double()).unwrap(),
            "i32" => FromPrimitive::from_i32(self.to_int()).unwrap(),
            "i64" => FromPrimitive::from_i64(self.to_long()).unwrap(),
            _ => todo!(),
        }
    }

    #[inline(always)]
    fn to_int(&self) -> i32 {
        match self.v {
            V::i(val) => checked_convert(val, "int"),
            V::f(val) => checked_convert(val, "int"),
        }
    }

    #[inline(always)]
    fn to_long(&self) -> i64 {
        match self.v {
            V::i(val) => checked_convert(val, "long"),
            V::f(val) => checked_convert(val, "long"),
        }
    }

    #[inline(always)]
    fn to_float(&self) -> f32 {
        match self.v {
            V::i(val) => checked_convert(val, "float"),
            V::f(val) => checked_convert(val, "float"),
        }
    }
    #[inline(always)]
    fn to_double(&self) -> f64 {
        match self.v {
            V::i(val) => checked_convert(val, "double"),
            V::f(val) => checked_convert(val, "double"),
        }
    }
    #[inline(always)]
    pub fn is_floating_point(&self) -> bool {
        match self.v {
            V::f(_) => true,
            _ => false,
        }
    }
    #[inline(always)]
    pub fn is_integer(&self) -> bool {
        match self.v {
            V::i(_) => true,
            _ => false,
        }
    }
}

impl From<f32> for Scalar {
    fn from(v: f32) -> Self {
        Self::new(v)
    }
}

impl From<f64> for Scalar {
    fn from(v: f64) -> Self {
        Self::new(v)
    }
}

impl From<i32> for Scalar {
    fn from(v: i32) -> Self {
        Self::new(v)
    }
}

impl From<usize> for Scalar {
    fn from(v: usize) -> Self {
        Self::new(v)
    }
}

impl From<Scalar> for f32 {
    fn from(s: Scalar) -> Self {
        s.to()
    }
}

impl From<Scalar> for usize {
    fn from(s: Scalar) -> Self {
        s.to()
    }
}
