use rovo::c10::TensorOptions;
use rovo::core::manual_seed;
use rovo::init_rovo;
use rovo::nn::Linear;
use rovo::nn::Module;
use rovo::tensor::loss::Reduction;
use rovo::{autograd, tensor::binary_cross_entropy};
use rovo::{autograd::backward, tensor::sigmoid};

#[test]
fn linear_backward_test() {
    init_rovo();
    manual_seed(0);
    let linear = Linear::new(4, 3);
    let x = autograd::full(&[2, 4], 3.0, TensorOptions::with_requires_grad());
    let y = linear.forward(&[&x]);
    // Expected: [[-2.0227153, 0.6529779, -0.6904765],[ -2.0227153, 0.6529779, -0.6904765]]
    println!("Result: {:?}", y);
    //Expected : -0.686738
    println!("Mean: {:?}", y.mean());

    backward::backward(&vec![y], &vec![], false);

    //Expected : [
    //           [-0.04011544, 0.08910112, -0.09542262, -0.011634579],
    //           [-0.04011544, 0.08910112, -0.09542262, -0.011634579]
    //          ]
    println!("Input Grad: {:?}", x.grad());
}

#[test]
fn linear_bce_test() {
    init_rovo();
    manual_seed(0);
    let linear = Linear::new(4, 3);
    let x = autograd::full(&[2, 4], 1.5, TensorOptions::with_requires_grad());
    let target = autograd::full(&[2, 3], 1.0, None);
    let y = linear.forward(&[&x]);
    let y = sigmoid(&y);
    let loss = binary_cross_entropy(&y, &target, None, Reduction::Mean);
    println!("Loss: {:?}", loss);
    backward(&vec![loss], &vec![], false);
    println!("Input Grad: {:?}", x.grad());
}
