use crate::aten::numeric_utils::IsNaN;
use std::marker::PhantomData;

pub trait SharedOps<A> {
    type ProjectArg;
    type ProjectRt;
    type ReduceArg1;
    type ReduceArg2;
    type CombineArg1;
    type CombineArg2;
    fn reduce(a: Self::ReduceArg1, b: Self::ReduceArg2, _idx: usize) -> A;
    fn combine(a: Self::CombineArg1, b: Self::CombineArg2) -> A;
    fn project(&self, a: Self::ProjectArg) -> Self::ProjectRt;
    fn translate_idx(acc: A, _base_dix: usize) -> A;
}

pub struct MeanOps<A, F> {
    factor: F,
    _ph: PhantomData<A>,
}

impl<A, F> MeanOps<A, F> {
    pub fn new(factor: F) -> Self {
        Self {
            factor,
            _ph: PhantomData,
        }
    }
}

impl<A, F> SharedOps<A> for MeanOps<A, F>
where
    A: std::ops::Add<Output = A>,
    F: std::ops::Mul<A, Output = A> + Copy,
{
    type ProjectArg = A;
    type ReduceArg1 = A;
    type ReduceArg2 = A;
    type CombineArg1 = A;
    type CombineArg2 = A;
    type ProjectRt = A;

    #[inline(always)]
    fn reduce(a: A, b: A, _idx: usize) -> A {
        Self::combine(a, b)
    }

    #[inline(always)]
    fn combine(a: A, b: A) -> A {
        a + b
    }

    #[inline(always)]
    fn project(&self, a: A) -> A {
        self.factor * a
    }

    #[inline(always)]
    fn translate_idx(acc: A, _base_idx: usize) -> A {
        return acc;
    }
}

trait CompareOps<T> {
    fn call(a: T, b: T, idx_a: usize, idx_b: usize) -> bool;
}

struct GreaterOrNan<T>(PhantomData<T>);

impl<T: IsNaN + PartialOrd> GreaterOrNan<T> {
    fn call(a: T, b: T, idx_a: usize, idx_b: usize) -> bool {
        if a.is_nan() {
            if b.is_nan() {
                return idx_a < idx_b;
            }
            return true;
        }
        if a == b {
            idx_a < idx_b
        } else {
            a > b
        }
    }
}

struct MinMaxReductionOps<A, C>(PhantomData<A>, PhantomData<C>);

impl<A, C> MinMaxReductionOps<A, C> {
    pub fn new() -> Self {
        Self(PhantomData, PhantomData)
    }
}

pub struct ArgMaxOps<T: IsNaN + PartialOrd> {
    _ph: PhantomData<T>,
}
impl<T: IsNaN + PartialOrd> ArgMaxOps<T> {
    pub fn new() -> Self {
        Self { _ph: PhantomData }
    }
}
impl<T: IsNaN + PartialOrd + Copy> SharedOps<(T, usize)> for ArgMaxOps<T> {
    type ProjectArg = (T, usize);

    type ReduceArg1 = (T, usize);

    type ReduceArg2 = T;

    type CombineArg1 = (T, usize);

    type CombineArg2 = (T, usize);

    type ProjectRt = usize;

    fn reduce(a: Self::ReduceArg1, b: Self::ReduceArg2, idx: usize) -> (T, usize) {
        if GreaterOrNan::<T>::call(a.0, b, a.1, idx) {
            a
        } else {
            (b, idx)
        }
    }

    fn combine(a: Self::CombineArg1, b: Self::CombineArg2) -> (T, usize) {
        if GreaterOrNan::<T>::call(a.0, b.0, a.1, b.1) {
            a
        } else {
            b
        }
    }

    fn project(&self, a: Self::ProjectArg) -> Self::ProjectRt {
        a.1
    }

    fn translate_idx(acc: (T, usize), base_idx: usize) -> (T, usize) {
        (acc.0, acc.1 + base_idx)
    }
}
