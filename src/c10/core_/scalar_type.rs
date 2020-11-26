use crate::c10::TypeMeta;
use std::mem::size_of;
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ScalarType {
    Byte,
    Char,
    Short,
    Long,
    Half,
    ComplexHalf,
    ComplexFloat,
    ComplexDouble,
    Bool,
    BFloat16,
    Int,
    Float,
    Double,
    QInt8,
    QUInt8,
    QInt32,
    Undefined,
    NumOptions,
}

impl Default for ScalarType {
    fn default() -> Self {
        ScalarType::Undefined
    }
}

impl From<ScalarType> for TypeMeta {
    fn from(s: ScalarType) -> Self {
        match s {
            ScalarType::Int => TypeMeta::make::<i32>(),
            ScalarType::Float => TypeMeta::make::<f32>(),
            ScalarType::Double => TypeMeta::make::<f64>(),
            ScalarType::Long => TypeMeta::make::<i64>(),
            _ => todo!(),
        }
    }
}

#[inline(always)]
pub fn type_meta_to_scalar_type(dtype: &TypeMeta) -> ScalarType {
    if let Some(scalar_type) = try_type_meta_to_scalar_type(dtype) {
        scalar_type
    } else {
        panic!("Unsupported TypeMeta in ATen");
    }
}

pub fn try_type_meta_to_scalar_type(dtype: &TypeMeta) -> Option<ScalarType> {
    if dtype == TypeMeta::make::<f32>() {
        return Some(ScalarType::Float);
    }
    if dtype == TypeMeta::make::<i32>() {
        return Some(ScalarType::Int);
    }
    return None;
}

static mut DEFAULT_DTYPE: Option<TypeMeta> = None;
pub fn get_default_dtype() -> TypeMeta {
    unsafe {
        if DEFAULT_DTYPE.is_none() {
            DEFAULT_DTYPE = Some(TypeMeta::make::<f32>())
        }
        DEFAULT_DTYPE.unwrap()
    }
}

#[inline(always)]
pub fn is_qint_type(t: ScalarType) -> bool {
    t == ScalarType::QInt8 || t == ScalarType::QUInt8 || t == ScalarType::QInt32
}
#[inline(always)]
pub fn is_floating_type(t: ScalarType) -> bool {
    t == ScalarType::Double
        || t == ScalarType::Float
        || t == ScalarType::Half
        || t == ScalarType::BFloat16
}
#[inline(always)]
pub fn is_complex_type(t: ScalarType) -> bool {
    t == ScalarType::ComplexHalf || t == ScalarType::ComplexFloat || t == ScalarType::ComplexDouble
}
#[inline(always)]
pub fn is_intgeral_type(t: ScalarType, included_bool: bool) -> bool {
    let is_integral = t == ScalarType::Byte
        || t == ScalarType::Char
        || t == ScalarType::Int
        || t == ScalarType::Long
        || t == ScalarType::Short;

    if included_bool {
        is_integral || t == ScalarType::Bool
    } else {
        is_integral
    }
}

#[inline(always)]
pub fn can_cast(from: ScalarType, to: ScalarType) -> bool {
    if is_complex_type(from) && !is_complex_type(to) {
        return false;
    }
    // We disallow float -> integral, e.g., int_tensor *= float is disallowed.
    if is_floating_type(from) && is_intgeral_type(to, false) {
        return false;
    }

    // Treat bool as a distinct "category," to be consistent with type promotion
    // rules (e.g. `bool_tensor + 5 -> int64_tensor`). If `5` was in the same category
    // as `bool_tensor`, we would not promote.
    // Differing categories implies `bool_tensor += 5` is disallowed.
    //
    // NB: numpy distinguishes "unsigned" as a category to get the desired
    // `bool_tensor + 5 -> int64_tensor` behavior. We don't, because:
    // * We don't want the performance hit of checking the runtime sign of Scalars.
    // * `uint8_tensor + 5 -> int64_tensor` would be undesirable.
    if from != ScalarType::Bool && to == ScalarType::Bool {
        return false;
    }
    return true;
}

#[macro_export]
macro_rules! AT_FORALL_SCALAR_TYPES_AND2 {
    // ($SCALARTYPE1: path, $SCALARTYPE2: path, $name: literal)=>{

    // }
    ($name: tt) => {
        $name!(u8, Byte);
        $name!(i32, Int);
        $name!(f32, Float);
    };
}

#[inline(always)]
pub fn elementSize(t: ScalarType) -> usize {
    match t {
        ScalarType::Byte => size_of::<u8>(),
        ScalarType::Char => size_of::<char>(),
        ScalarType::Bool => size_of::<bool>(),
        ScalarType::Int => size_of::<i32>(),
        ScalarType::Float => size_of::<f32>(),
        _ => todo!(),
    }
}
