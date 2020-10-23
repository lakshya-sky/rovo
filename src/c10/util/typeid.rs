#[derive(Debug, Copy, Clone)]
pub struct TypeMeta {
    itemsize: usize,
}

impl TypeMeta {
    pub fn itemsize(&self) -> usize {
        self.itemsize
    }
    pub fn make<T>() -> Self {
        let itemsize = std::mem::size_of::<T>();
        Self { itemsize }
    }
}
