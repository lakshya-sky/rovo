#[inline(always)]
pub fn checked_convert<T: num::NumCast, U: num::NumCast>(v: T, _name: &str) -> U {
    num::cast(v).unwrap()
}
