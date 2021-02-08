#[inline(always)]
pub fn checked_convert<T: num::NumCast, U: num::NumCast>(v: T, _name: &str) -> U {
    num::cast(v).unwrap()
}

pub fn cast_with_inter_type<I: num::NumCast, O: num::NumCast>(v: I) -> O {
    num::cast(v).unwrap()
}
