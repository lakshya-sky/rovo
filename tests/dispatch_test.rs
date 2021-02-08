use rovo::c10::ScalarType;
use rovo::AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2;

#[test]
fn test_at_private_case_type_macro() {
    let type_ = ScalarType::Float;
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2!(_, _, type_, "test_dispatch", || {
        println!("Float Type")
    })
}
