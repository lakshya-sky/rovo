#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum DeviceType {
    Cpu,
    Cuda,
    COMPILE_TIME_MAX_DEVICE_TYPE,
}
pub const KCPU: DeviceType = DeviceType::Cpu;
pub const KCUDA: DeviceType = DeviceType::Cuda;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Device {
    type_: DeviceType,
    index: i16,
}

impl Default for Device {
    fn default() -> Self {
        Self {
            type_: DeviceType::Cpu,
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
    pub fn type_(&self) -> DeviceType {
        self.type_
    }

    pub fn is_cpu(&self) -> bool {
        self == DeviceType::Cpu
    }
    /// Returns true if the device has a non-default index.
    pub fn has_index(&self) -> bool {
        self.index != -1
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

impl PartialEq<DeviceType> for &Device {
    fn eq(&self, other: &DeviceType) -> bool {
        let other: Device = other.into();
        *self == &other
    }
}
