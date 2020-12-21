use rovo::{autograd, tensor::log_softmax};
use rovo::{c10::TensorOptions, init_rovo};

#[test]
fn empty_tensor_and_fill_ones() {
    rovo::init_rovo();
    let t = autograd::empty(&[2, 2], None, None);
    t.fill_(1.0);
    println!("{:?}", t);
}

#[test]
fn test_fill_and_add() {
    rovo::init_rovo();
    let t = autograd::empty(&[2, 2], None, None);
    let r = autograd::empty(&[2, 2], None, None);
    t.fill_(1.0);
    r.fill_(1.23);
    t.add_(&r);
    println!("{:?}", t);
}

#[test]
fn test_neg() {
    rovo::init_rovo();
    let t = autograd::full(&[2, 2], 1.0, TensorOptions::with_requires_grad());
    let r = -&t;
    println!("{:?}", r);
}

#[test]
fn test_div_scalar() {
    rovo::init_rovo();
    let mut x = autograd::full(&[2, 2], 3.0, TensorOptions::with_requires_grad());
    x.div_scalar(2.0);
    println!("test_div_scalar_result: {:?}", x);
}

#[test]
fn test_add_scalar() {
    rovo::init_rovo();
    let x = autograd::full(&[2, 3], 3.0, TensorOptions::with_requires_grad());
    let result = x + 2.0;
    println!("test_add_scalar_result: {:?}", result);
}

#[test]
fn test_t() {
    rovo::init_rovo();
    let t = autograd::full(&[2, 3], 1.0, TensorOptions::with_requires_grad());
    let r = t.t();
    assert_eq!(r.sizes(), &[3, 2]);
    println!("{:?}", r);
}
#[test]
fn test_mm() {
    rovo::init_rovo();
    let x = autograd::full(&[2, 2], 3.0, TensorOptions::with_requires_grad());
    let w = autograd::full(&[2, 1], 2.0, TensorOptions::with_requires_grad());
    let result = x.mm(&w, true);
    println!("Result: {:?}", result);
    autograd::backward(&vec![result], &vec![], false);
    println!("{:?}", x.grad().unwrap().as_ref());
    println!("{:?}", w.grad().unwrap().as_ref());
}
#[test]
fn test_blas() {
    let (m, n, k) = (2, 1, 2);
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![3.0, 4.0];
    let mut c = vec![0.0, 0.0];
    unsafe {
        blas_sys::sgemm_(
            &(b'T' as i8),
            &(b'T' as i8),
            &m,
            &n,
            &k,
            &1.0,
            a.as_ptr(),
            &m,
            b.as_ptr(),
            &n,
            &0.0,
            c.as_mut_ptr(),
            &k,
        )
    }
    dbg!(c);
}

#[test]
fn logsoftmax() {
    init_rovo();
    let x = autograd::full(&[5, 5], 1.7, TensorOptions::with_requires_grad());
    // let t = autograd::full(&[5, 1], 1.0, TensorOptions::with_requires_grad());
    let result = log_softmax(&x, 1, None);
    println!("Logsoftmax result: {:?}", result);
}
