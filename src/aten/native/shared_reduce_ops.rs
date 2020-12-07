use std::marker::PhantomData;

pub trait SharedOps<A> {
    type ProjectArg;
    type ReduceArg1;
    type ReduceArg2;
    type CombineArg1;
    type CombineArg2;
    fn reduce(a: A, b: A, _idx: usize) -> A;
    fn combine(a: A, b: A) -> A;
    fn project(&self, a: A) -> A;
    fn translate_idx(acc: A, _base_dix: usize) -> A;
}

pub struct MeanOps<A, F> {
    factor: F,
    _phtm: PhantomData<A>,
}

impl<A, F> MeanOps<A, F> {
    pub fn new(factor: F) -> Self {
        Self {
            factor,
            _phtm: PhantomData,
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

/*inline C10_DEVICE acc_t reduce(acc_t a, acc_t b, int64_t /*idx*/) const {
    return combine(a, b);
  }

  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }

  inline C10_DEVICE acc_t project(acc_t a) const {
    return a * factor;
  }

  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline C10_DEVICE acc_t warp_shfl_down(acc_t data, int offset) const {
    return WARP_SHFL_DOWN(data, offset);
  }
#endif

  MeanOps(factor_t factor): factor(factor) {
  }
};
*/
