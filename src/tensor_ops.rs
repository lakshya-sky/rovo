use crate::variable::*;
use std::ops::Add;

impl Add<Self> for Tensor {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let result = &self._impl.borrow().data + &rhs._impl.borrow().data;
        let _impl = TensorImpl {
            data: result,
            autogradmeta: None,
        };
        Tensor::from_impl(_impl)
    }
}

impl Add<Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Self::Output {
        let result = &self._impl.borrow().data + &rhs._impl.borrow().data;
        let _impl = TensorImpl {
            data: result,
            autogradmeta: None,
        };
        Tensor::from_impl(_impl)
    }
}

// impl AddAssign<Tensor> for Tensor {
//     fn add_assign(&mut self, rhs: Tensor) {
//         *self = Self{

//         }
//     }
// }
