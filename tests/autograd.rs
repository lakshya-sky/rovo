use rovo::{aten::native::argmax, autograd::tensor, c10::TensorOptions, init_rovo};

#[test]
fn tensor_from_slice() {
    init_rovo();
    let tensor = tensor(
        &[1.1f32, 2.1, 3.1, 4.1, 5.1, 6.1],
        TensorOptions::with_requires_grad(),
    );
    println!("{:?}", tensor);
}
#[test]
fn tensor_from_i64_slice() {
    init_rovo();
    let tensor = tensor(&[1i64, 2, 3], TensorOptions::with_requires_grad());
    println!("{:?}", tensor);
}

#[test]
fn test_argmax() {
    init_rovo();
    let tensor = tensor(
        &[1.1f32, 2.1, 3.1, 4.1, 5.1, 6.1],
        TensorOptions::with_requires_grad(),
    );
    println!("{:?}", argmax(&tensor, None, false));
}
