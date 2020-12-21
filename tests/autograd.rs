use rovo::{aten::native::tensor, c10::TensorOptions, init_rovo};

#[test]
fn tensor_from_slice() {
    init_rovo();
    let tensor = tensor(
        &[1.1f32, 2.1, 3.1, 4.1, 5.1, 6.1],
        TensorOptions::with_requires_grad(),
    );
    println!("{:?}", tensor);
}