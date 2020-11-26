pub fn normal<V>(val: V, mean: V, std: V) -> V
where
    V: std::ops::Mul<Output = V>,
    V: std::ops::Add<Output = V>,
{
    (val * std) + mean
}

pub fn uniform_real<
    T: num::cast::NumCast
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + Copy,
    V: num::cast::NumCast + std::ops::BitAnd<Output = V> + std::ops::Mul<Output = V> + Copy,
>(
    val: V,
    from: T,
    to: T,
) -> T {
    let mask: V;
    let divisor: T;
    if std::any::type_name::<T>() == "f64" {
        mask = num::cast((1u64 << std::f64::MANTISSA_DIGITS) - 1).unwrap();
        divisor = num::cast::<_, T>(1.0 / (1u64 << std::f64::MANTISSA_DIGITS) as f64).unwrap();
    } else {
        mask = num::cast((1u64 << std::f32::MANTISSA_DIGITS) - 1).unwrap();
        divisor = num::cast::<_, T>(1.0f32 / (1u64 << std::f32::MANTISSA_DIGITS) as f32).unwrap();
    }
    let tmp = val & mask;
    let x: T = num::cast::<V, T>(tmp).unwrap() * divisor;
    let result = x * (to - from) + from;
    result
}
