mod loops;
pub use loops::*;

mod tensor_compare;
pub use tensor_compare::*;

pub mod cpu;
pub mod distribution_templates;

mod tensor_factories;
pub use tensor_factories::*;

mod fill;
pub use fill::*;

mod resize;
pub use resize::*;

mod binary_ops;
pub use binary_ops::*;

use crate::c10::*;
use crate::tensor::NewTensor;
#[derive(Default, Clone)]
pub struct ResultTypeState {
    dim_result: ScalarType,
    wrapped_result: ScalarType,
    zero_result: ScalarType,
}

pub fn update_result_type_state(tensor: &NewTensor, in_state: ResultTypeState) -> ResultTypeState {
    if !tensor.defined() {
        return in_state;
    }
    let mut new_state = in_state.clone();
    let mut current = tensor.scalar_type();
    if tensor.get_unsafe_tensor_impl().is_wrapped_number() {
        let current_default = type_meta_to_scalar_type(&get_default_dtype());
        if is_floating_type(current) {
            current = current_default;
        }
    }
    if tensor.dim() > 0 {
        new_state.dim_result = promote_skip_undefined(in_state.dim_result, current);
    } else if tensor.get_unsafe_tensor_impl().is_wrapped_number() {
        new_state.wrapped_result = promote_skip_undefined(in_state.wrapped_result, current);
    } else {
        new_state.zero_result = promote_skip_undefined(in_state.zero_result, current);
    }
    new_state
}

#[inline(always)]
fn promote_skip_undefined(a: ScalarType, b: ScalarType) -> ScalarType {
    if a == ScalarType::Undefined {
        return b;
    }
    if b == ScalarType::Undefined {
        return a;
    }
    promote_types(a, b)
}

#[inline(always)]
fn promote_types(a: ScalarType, b: ScalarType) -> ScalarType {
    // This is generated according to NumPy's promote_types
    let u1 = ScalarType::Byte;
    let i1 = ScalarType::Char;
    let i2 = ScalarType::Short;
    let i4 = ScalarType::Int;
    let i8 = ScalarType::Long;
    let f2 = ScalarType::Half;
    let f4 = ScalarType::Float;
    let f8 = ScalarType::Double;
    let c2 = ScalarType::ComplexHalf;
    let c4 = ScalarType::ComplexFloat;
    let c8 = ScalarType::ComplexDouble;
    let b1 = ScalarType::Bool;
    let bf = ScalarType::BFloat16;
    let ud = ScalarType::Undefined;
    if a == ud || b == ud {
        return ScalarType::Undefined;
    }

    // For QInt types, we only allow exact match
    if is_qint_type(a) && a == b {
        return a;
    }

    if is_qint_type(a) || is_qint_type(b) {
        panic!(
          "promoteTypes with quantized numbers is not handled yet; figure out what the correct rules should be, offending types: {:?} {:?}",
          a,
          b);
    }

    // this matrix has to be consistent with AT_FORALL_SCALAR_TYPES_WITH_COMPLEX
    // so that's why we have to add undefined as we are not sure what is the
    // corrent values for the type promotions in complex type cases.
    let promote_types_lookup = [
        /*        u1  i1  i2  i4  i8  f2  f4  f8  c2  c4  c8  b1  q1  q2  q3  bf*/
        /* u1 */
        [
            u1, i2, i2, i4, i8, f2, f4, f8, ud, c4, c8, u1, ud, ud, ud, ud,
        ],
        /* i1 */
        [
            i2, i1, i2, i4, i8, f2, f4, f8, ud, c4, c8, i1, ud, ud, ud, ud,
        ],
        /* i2 */
        [
            i2, i2, i2, i4, i8, f2, f4, f8, ud, c4, c8, i2, ud, ud, ud, ud,
        ],
        /* i4 */
        [
            i4, i4, i4, i4, i8, f2, f4, f8, ud, c4, c8, i4, ud, ud, ud, ud,
        ],
        /* i8 */
        [
            i8, i8, i8, i8, i8, f2, f4, f8, ud, c4, c8, i8, ud, ud, ud, ud,
        ],
        /* f2 */
        [
            f2, f2, f2, f2, f2, f2, f4, f8, ud, c4, c8, f2, ud, ud, ud, ud,
        ],
        /* f4 */
        [
            f4, f4, f4, f4, f4, f4, f4, f8, ud, c4, c8, f4, ud, ud, ud, ud,
        ],
        /* f8 */
        [
            f8, f8, f8, f8, f8, f8, f8, f8, ud, c8, c8, f8, ud, ud, ud, ud,
        ],
        /* c2 */
        [
            ud, ud, ud, ud, ud, ud, ud, ud, c2, c4, c8, ud, ud, ud, ud, ud,
        ],
        /* c4 */
        [
            c4, c4, c4, c4, c4, c4, c4, c8, c4, c4, c8, c4, ud, ud, ud, ud,
        ],
        /* c8 */
        [
            c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, ud, ud, ud, ud,
        ],
        /* b1 */
        [
            u1, i1, i2, i4, i8, f2, f4, f8, ud, c4, c8, b1, ud, ud, ud, ud,
        ],
        /* q1 */
        [
            ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud,
        ],
        /* q2 */
        [
            ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud,
        ],
        /* q3 */
        [
            ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud,
        ],
        /* bf */
        [
            ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, bf,
        ],
    ];
    return promote_types_lookup[a as usize][b as usize];
}

pub fn result_type(in_state: &ResultTypeState) -> ScalarType {
    combine_categories(
        in_state.dim_result,
        combine_categories(in_state.zero_result, in_state.wrapped_result),
    )
}

#[inline(always)]
fn combine_categories(higher: ScalarType, lower: ScalarType) -> ScalarType {
    if is_complex_type(higher) {
        return higher;
    } else if !is_complex_type(lower) && is_floating_type(higher) {
        return higher;
    }
    if higher == ScalarType::Bool || is_floating_type(lower) || is_complex_type(lower) {
        return promote_skip_undefined(higher, lower);
    }
    if higher != ScalarType::Undefined {
        return higher;
    }
    return lower;
}
