use crate::{c10::DeviceType, tensor::TensorIterator};

pub trait DispatchStub {
    fn call(&self, device_type: DeviceType, iter: TensorIterator);
}
