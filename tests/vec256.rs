// use num::Float;
// use rovo::{
//     core::{
//         mt19937_engine::MT19937Engine, read_random_long, NormalDistribution,
//         UniformRealDistribution,
//     },
//     util::vec256::*,
// };
// use std::ffi::c_void;
// struct TestSeed(u64);
// impl Default for TestSeed {
//     fn default() -> Self {
//         Self(read_random_long())
//     }
// }
// impl TestSeed {
//     pub fn new(seed: u64) -> Self {
//         Self(seed)
//     }
//     pub fn get(&self) -> u64 {
//         self.0
//     }

//     pub fn add(&self, index: u64) -> Self {
//         Self::new(self.get() + index)
//     }
// }
// trait ValueGen {
//     type Output;
//     fn get(&self) -> Self::Output;
// }

// struct FloatValueGen<T: Float + From<f32>> {
//     gen: MT19937Engine,
//     normal: NormalDistribution<T>,
//     round_chance: UniformRealDistribution<i32>,
//     start: T,
//     end: T,
//     use_sign_change: bool,
//     use_round: bool,
// }

// impl<T: Float + From<f32>> Default for FloatValueGen<T> {
//     fn default() -> Self {
//         Self::new(T::min_value(), T::max_value())
//     }
// }

// impl<T: Float + From<f32>> FloatValueGen<T> {
//     pub fn new(start: T, end: T) -> Self {
//         let seed = TestSeed::default().get();
//         Self::with_seed(start, end, seed)
//     }

//     pub fn with_seed(start: T, end: T, seed: u64) -> Self {
//         let gen = MT19937Engine::new(Some(seed));
//         let t: T = (0.5).into();
//         let mean = start * (0.5).into() + end * (0.5).into();
//         let div_range: T = (0.6).into();
//         let stdev = T::abs_sub(end / div_range, start / div_range);
//         let normal = NormalDistribution::new(mean, stdev);
//         let round_chance = UniformRealDistribution::new(0, 5);
//         Self {
//             start,
//             end,
//             use_sign_change: false,
//             use_round: true,
//             round_chance,
//             normal,
//             gen,
//         }
//     }
// }

// fn local_exp<T: Float>(x: T) -> T {
//     x.exp()
// }

// fn local_add<T: num::Num>(x: T, y: T) -> T {
//     x + y
// }

// struct TestingCase<T>(T);
// fn test_unary<T: Float, F1, F2>(
//     testNameInfo: &str,
//     expectedFunction: F1,
//     actualFunction: F2,
// ) -> (Vec256<T>, Vec256<T>)
// where
//     F1: Fn(T) -> T,
//     F2: for<'a> Fn(&'a Vec256<T>) -> Vec256<T>,
// {
//     let el_count = Vec256::<T>::size();
//     let mut vals = Vec::<T>::with_capacity(el_count);
//     let mut expected = Vec::<T>::with_capacity(el_count);
//     for k in 0..el_count {
//         vals.push(T::one());
//         expected.push(expectedFunction(vals[k]));
//     }
//     let input = Vec256::<T>::loadu(vals.as_ptr() as *const c_void, el_count);
//     let actual = actualFunction(&input);
//     let vec_expected = Vec256::<T>::loadu(expected.as_ptr() as *const c_void, el_count);
//     (actual, vec_expected)
// }

// #[test]
// fn test_local_exp() {
//     type vec = Vec256<f32>;
//     let actual_fn = |v1: &vec| -> vec { v1.exp() };
//     let (a, e) = test_unary::<f32, _, _>("exp", local_exp, actual_fn);
// }
