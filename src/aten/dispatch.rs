use crate::c10::ScalarType;

#[macro_export]
macro_rules! AT_PRIVATE_CASE_TYPE{
    ($_ident: expr, $enum_type: path, $type: ty, $($args:expr),+)=>{
       if let $enum_type = $_ident {
            type SCALART = $type;
            return $($args),+()
        }
    }
}

#[macro_export]
macro_rules! AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2 {
    ($TYPE: expr, $name: expr, $($args:expr),+) => {{
        // match $TYPE {
        //     ScalarType::Int => {
        //         type SCALART = i32;
        //         $($args)*()
        //     },
        //     _ => todo!()
        // };
        AT_PRIVATE_CASE_TYPE!($TYPE, ScalarType::Float, f32, $($args)+);
        AT_PRIVATE_CASE_TYPE!($TYPE, ScalarType::Int, i32, $($args)+);
        AT_PRIVATE_CASE_TYPE!($TYPE, ScalarType::Double, f64, $($args)+);
    }};
}
