use crate::c10::{
    get_default_dtype, Device, DeviceType, Layout, MemoryFormat, ScalarType, TypeMeta, K_STRIDED,
};
#[derive(Clone)]
pub struct TensorOptions {
    dtype: TypeMeta,
    device: Device,
    layout: Layout,
    memory_format: MemoryFormat,
    requires_grad: bool,
    pinned_memory: bool,
    has_device: bool,
    has_dtype: bool,
    has_layout: bool,
    has_requires_grad: bool,
    has_pinned_memory: bool,
    has_memory_format: bool,
}

impl Default for TensorOptions {
    fn default() -> Self {
        Self {
            requires_grad: false,
            pinned_memory: false,
            has_device: false,
            has_dtype: false,
            has_layout: false,
            has_requires_grad: false,
            has_pinned_memory: false,
            has_memory_format: false,
            dtype: TypeMeta::make::<f32>(),
            device: Device::default(),
            layout: K_STRIDED,
            memory_format: MemoryFormat::Contiguous,
        }
    }
}

impl TensorOptions {
    pub fn with_layout(layout: Layout) -> Self {
        Self {
            layout,
            ..Self::default()
        }
    }
    pub fn with_device_type(type_: DeviceType, index: Option<i16>) -> Self {
        let device = Device::new(type_, index);
        Self::with_device(device)
    }
    pub fn with_device(device: Device) -> Self {
        Self {
            device,
            ..Self::default()
        }
    }
    pub fn with_dtype(dtype: TypeMeta) -> Self {
        Self {
            dtype,
            ..Self::default()
        }
    }
    pub fn with_memory_format(memory_format: MemoryFormat) -> Self {
        Self {
            memory_format,
            ..Self::default()
        }
    }

    pub fn set_dtype<T: Into<Option<TypeMeta>>>(&self, dtype: T) -> Self {
        let mut clone = self.clone();
        clone.set_dtype_mut(dtype);
        clone
    }

    pub fn set_dtype_<T: Into<Option<ScalarType>>>(&self, scalar_type: T) -> Self {
        let mut clone = self.clone();
        let dtype: Option<TypeMeta> = scalar_type.into().map_or(None, |s| Some(s.into()));
        clone.set_dtype_mut(dtype);
        clone
    }

    pub fn set_dtype_mut<T: Into<Option<TypeMeta>>>(&mut self, dtype: T) {
        if let Some(dtype) = dtype.into() {
            self.dtype = dtype;
            self.has_dtype = true;
        } else {
            self.has_dtype = false;
        }
    }

    pub fn set_device<T: Into<Option<Device>>>(&self, device: T) -> Self {
        let mut clone = self.clone();
        clone.set_device_mut(device);
        clone
    }

    pub fn set_device_mut<T: Into<Option<Device>>>(&mut self, device: T) {
        if let Some(device) = device.into() {
            self.device = device;
            self.has_device = true;
        } else {
            self.has_device = false;
        }
    }

    pub fn set_layout<T: Into<Option<Layout>>>(&self, layout: T) -> Self {
        let mut clone = self.clone();
        clone.set_layout_mut(layout);
        clone
    }

    pub fn set_layout_mut<T: Into<Option<Layout>>>(&mut self, layout: T) {
        if let Some(layout) = layout.into() {
            self.layout = layout;
            self.has_layout = true;
        } else {
            self.has_layout = false;
        }
    }

    pub fn with_requires_grad() -> Self {
        Self::with_requires_grad_(true)
    }

    pub fn with_requires_grad_(requires_grad: bool) -> Self {
        let mut o = Self::default();
        o.set_requires_grad_mut(requires_grad);
        o
    }

    pub fn requires_grad(&self) -> bool {
        if self.has_requires_grad {
            self.requires_grad
        } else {
            false
        }
    }

    pub fn set_requires_grad<T: Into<Option<bool>>>(&self, requires_grad: T) -> TensorOptions {
        let mut clone = self.clone();
        clone.set_requires_grad_mut(requires_grad);
        clone
    }
    pub fn set_requires_grad_mut<T: Into<Option<bool>>>(&mut self, requires_grad: T) {
        if let Some(requires_grad) = requires_grad.into() {
            self.requires_grad = requires_grad;
            self.has_requires_grad = true;
        } else {
            self.has_requires_grad = false;
        }
    }

    pub fn dtype(&self) -> TypeMeta {
        if self.has_device {
            self.dtype
        } else {
            get_default_dtype()
        }
    }

    pub fn has_device(&self) -> bool {
        self.has_device
    }
    pub fn has_dtype(&self) -> bool {
        self.has_dtype
    }
    pub fn has_layout(&self) -> bool {
        self.has_layout
    }
    pub fn has_requires_grad(&self) -> bool {
        self.has_requires_grad
    }
    pub fn device_opt(&self) -> Option<Device> {
        if self.has_device {
            Some(self.device.clone())
        } else {
            None
        }
    }

    pub fn device(&self) -> Device {
        if self.has_device {
            self.device.clone()
        } else {
            Device::new(DeviceType::CPU, None)
        }
    }

    pub fn layout_opt(&self) -> Option<Layout> {
        if self.has_layout {
            Some(self.layout)
        } else {
            None
        }
    }

    pub fn dtype_opt(&self) -> Option<TypeMeta> {
        if self.has_dtype {
            Some(self.dtype)
        } else {
            None
        }
    }

    pub fn requires_grad_opt(&self) -> Option<bool> {
        if self.has_requires_grad {
            Some(self.requires_grad)
        } else {
            None
        }
    }

    pub fn merge_in<A: AsRef<Self>>(&self, options: A) -> Self {
        let mut r = options.as_ref().clone();
        if !r.has_device() {
            r.set_device_mut(self.device_opt());
        }
        if !r.has_dtype() {
            r.set_dtype_mut(self.dtype_opt());
        }
        if !r.has_layout() {
            r.set_layout_mut(self.layout_opt());
        }
        if !r.has_requires_grad() {
            r.set_requires_grad_mut(self.requires_grad_opt());
        }
        // if !r.has_pinned_memory() r.set_pinned_memory(pinned_memory_opt());
        // if !r.has_memory_format() r.set_memory_format(memory_format_opt());
        r
    }
}

impl AsRef<Self> for TensorOptions {
    fn as_ref(&self) -> &Self {
        self
    }
}

pub fn device(device: impl Into<Device>) -> TensorOptions {
    TensorOptions::with_device(device.into())
}
