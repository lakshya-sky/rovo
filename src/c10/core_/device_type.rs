#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum DeviceType {
    CPU,
    CUDA,
    COMPILE_TIME_MAX_DEVICE_TYPE,
}
pub const KCPU: DeviceType = DeviceType::CPU;
pub const KCUDA: DeviceType = DeviceType::CUDA;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Device {
    type_: DeviceType,
    index: i16,
}

impl Default for Device {
    fn default() -> Self {
        Self {
            type_: DeviceType::CPU,
            index: -1,
        }
    }
}

impl Device {
    pub fn new(type_: DeviceType, index: Option<i16>) -> Self {
        Self {
            type_,
            index: index.unwrap_or(-1),
        }
    }
}

impl From<&DeviceType> for Device {
    fn from(device_type: &DeviceType) -> Self {
        Device::new(*device_type, None)
    }
}

impl From<DeviceType> for Device {
    fn from(device_type: DeviceType) -> Self {
        Device::new(device_type, None)
    }
}

impl PartialEq<DeviceType> for Device {
    fn eq(&self, other: &DeviceType) -> bool {
        let other: Device = other.into();
        self == &other
    }
}
