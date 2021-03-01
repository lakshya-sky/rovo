use rovo::{aten::native::empty, c10::TensorOptions, init_rovo};

#[test]
fn test_resize() {
    init_rovo();
    let a = empty(&[0], TensorOptions::default(), None);
    a.resize(&[3, 4], None);
    assert_eq!(a.numel(), 12);
    a.resize(&[5, 7], None);
    assert_eq!(a.numel(), 35);
}
