use once_cell::sync::OnceCell;
#[derive(Debug, Copy, Clone, Default)]
pub struct TypeMeta {
    data: Option<&'static TypeMetaData>,
}

impl TypeMeta {
    pub fn itemsize(&self) -> usize {
        self.data.as_ref().unwrap().itemsize
    }
    pub fn make<T>() -> Self {
        let data = match std::any::type_name::<T>() {
            "f32" => type_meta_data_instance_float(),
            "i32" => todo!(),
            _ => panic!(),
        };

        Self { data: Some(data) }
    }
}

impl std::cmp::PartialEq<TypeMeta> for &TypeMeta {
    fn eq(&self, other: &TypeMeta) -> bool {
        *self.data.as_ref().unwrap() as *const TypeMetaData
            == *other.data.as_ref().unwrap() as *const TypeMetaData
    }
}
#[derive(Debug, Copy, Clone)]
struct TypeMetaData {
    name: &'static str,
    itemsize: usize,
}

fn type_meta_data_instance_float() -> &'static TypeMetaData {
    static SINGLETON: OnceCell<TypeMetaData> = OnceCell::new();
    SINGLETON.get_or_init(|| make_type_meta_data_instance::<f32>())
}

fn make_type_meta_data_instance<T>() -> TypeMetaData {
    let typename = std::any::type_name::<T>();
    let itemsize = std::mem::size_of::<T>();
    TypeMetaData {
        name: typename,
        itemsize,
    }
}
