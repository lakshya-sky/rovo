use crate::aten::native;
use crate::c10::*;
use crate::tensor::Tensor;

#[inline(always)]
fn get_device<D: Into<Option<DeviceType>>>(d: D) -> DeviceType {
    match d.into() {
        None => DeviceType::CPU,
        Some(dd) => dd,
    }
}

#[inline(always)]
pub fn scalar_to_tensor<D: Into<Option<DeviceType>>>(s: Scalar, device_type: D) -> Tensor {
    let device_type = get_device(device_type);
    match device_type {
        DeviceType::CPU => {
            if s.is_floating_point() {
                native::scalar_tensor(s, device(device_type).set_dtype_(ScalarType::Double))
            } else if s.is_integer() {
                native::scalar_tensor(s, device(device_type).set_dtype_(ScalarType::Long))
            } else {
                todo!()
            }
        }
        _ => todo!(),
    }
}
