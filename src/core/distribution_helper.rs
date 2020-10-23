use super::{generator::GeneratorImpl, transformation_helper};

pub struct UniformRealDistribution<T> {
    from: T,
    to: T,
}
impl<T> UniformRealDistribution<T>
where
    T: Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Mul<Output = T>
        + num::cast::NumCast,
{
    pub fn new(from: T, to: T) -> Self {
        Self { from, to }
    }
    pub fn call(&self, gen: &dyn GeneratorImpl) -> T {
        if std::any::type_name::<T>() == "f64" {
            transformation_helper::uniform_real(gen.random64(), self.from, self.to)
        } else {
            transformation_helper::uniform_real(gen.random(), self.from, self.to)
        }
    }
}
