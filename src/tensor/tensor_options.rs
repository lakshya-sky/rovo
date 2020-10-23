use crate::c10::{Device, DeviceType, Layout, MemoryFormat, TypeMeta, K_STRIDED};
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

    // pub fn merge_in(&self, options: Self) -> Self {
    //     let mut r = options.clone();
    //     if !r.has_device() {
    //         r.set_device(device_opt());
    //     }
    //     if !r.has_dtype() {
    //         r.set_dtype(dtype_opt());
    //     }
    //     if !r.has_layout() {
    //         r.set_layout(layout_opt());
    //     }
    //     if !r.has_requires_grad() {
    //         r.set_requires_grad(requires_grad_opt());
    //     }
    //     // if !r.has_pinned_memory() r.set_pinned_memory(pinned_memory_opt());
    //     // if !r.has_memory_format() r.set_memory_format(memory_format_opt());
    //     r
    // }
}

static mut DEFAULT_DTYPE: Option<TypeMeta> = None;
fn get_default_dtype() -> TypeMeta {
    unsafe {
        if DEFAULT_DTYPE.is_none() {
            DEFAULT_DTYPE = Some(TypeMeta::make::<f32>())
        }
        DEFAULT_DTYPE.unwrap()
    }
}
