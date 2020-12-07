use once_cell::sync::OnceCell;
#[derive(Debug, Copy, Clone, Default)]
pub struct TypeMeta {
    data: Option<&'static TypeMetaData>,
}

impl TypeMeta {
    pub fn itemsize(&self) -> usize {
        self.data.as_ref().unwrap().itemsize
    }
    pub fn make<T: Trait>() -> TypeMeta {
        TypeMeta {
            data: Some(T::make()),
        }
    }
}

impl std::cmp::PartialEq<TypeMeta> for &TypeMeta {
    fn eq(&self, other: &TypeMeta) -> bool {
        *self.data.as_ref().unwrap() as *const TypeMetaData
            == *other.data.as_ref().unwrap() as *const TypeMetaData
    }
}
#[derive(Debug, Copy, Clone)]
pub struct TypeMetaData {
    name: &'static str,
    itemsize: usize,
}

pub trait Trait {
    fn make() -> &'static TypeMetaData;
}

impl Trait for f32 {
    fn make() -> &'static TypeMetaData {
        static SINGLETON: OnceCell<TypeMetaData> = OnceCell::new();
        SINGLETON.get_or_init(|| make_type_meta_data_instance::<f32>())
    }
}
impl Trait for i32 {
    fn make() -> &'static TypeMetaData {
        static SINGLETON: OnceCell<TypeMetaData> = OnceCell::new();
        SINGLETON.get_or_init(|| make_type_meta_data_instance::<i32>())
    }
}
impl Trait for f64 {
    fn make() -> &'static TypeMetaData {
        static SINGLETON: OnceCell<TypeMetaData> = OnceCell::new();
        SINGLETON.get_or_init(|| make_type_meta_data_instance::<f64>())
    }
}
impl Trait for i64 {
    fn make() -> &'static TypeMetaData {
        static SINGLETON: OnceCell<TypeMetaData> = OnceCell::new();
        SINGLETON.get_or_init(|| make_type_meta_data_instance::<i64>())
    }
}

fn make_type_meta_data_instance<T>() -> TypeMetaData {
    let typename = std::any::type_name::<T>();
    let itemsize = std::mem::size_of::<T>();
    TypeMetaData {
        name: typename,
        itemsize,
    }
}
