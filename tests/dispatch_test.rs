use rovo::c10::ScalarType;
use rovo::AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2;
use rovo::AT_PRIVATE_CASE_TYPE;

macro_rules! my_match {
    ($obj:expr, $($matcher:pat $(if $pred:expr)* => $result:expr),*) => {
        match $obj {
            $($matcher $(if $pred)* => $result),*
        }
    }
 }

#[test]
fn test_at_private_case_type_macro() {
    let type_ = ScalarType::Float;
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2!(type_, "test_dispatch", || { println!("Float Type") })
}

#[test]
fn test_my_macro() {
    let x = 7;
    let s = my_match! {
        x,
        10 => "Ten",
        n if x < 5 => "Less than 5",
        _ => "something else"
    };
    println!("s = {:?}", s);
}
