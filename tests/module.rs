use rovo::core::manual_seed;
use rovo::init_rovo;
use rovo::nn::Linear;
use rovo::nn::Module;
use rovo::optim::{Optimizer, SGDOptions, SGD};
use rovo::tensor::loss::Reduction;
use rovo::tensor::sigmoid;
use rovo::tensor::{binary_cross_entropy, Tensor};
use rovo::{
    autograd::{self, backward, full, ones},
    nn::Sequential,
};
use rovo::{c10::TensorOptions, nn::Functional};

#[test]
fn linear_parameter_test() {
    init_rovo();
    manual_seed(1);
    let mut model = Sequential::new();
    model.add(Linear::new(4, 3));
    dbg!(model.parameters());
    //Expected: [
    //     Tensor: [0.2576315999031067, -0.22068911790847778, -0.09693074226379395,
    // 0.23468446731567383, -0.4707184433937073, 0.29985862970352173, -0.1028626561164856,
    // 0.2543719410896301, 0.06950849294662476, -0.061222076416015625, 0.13868045806884766,
    // 0.024665892124176025]   size: [3, 4],
    //     Tensor: [0.18261408805847168, -0.19485050439834595, -0.03645437955856323]       size: [3],
    // ],
}

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
#[test]
fn sequential_sgd_step() {
    init_rovo();
    manual_seed(0);

    let x = full(&[4, 3], 1.5, TensorOptions::with_requires_grad());
    let target = ones(&[4, 2], None);
    let mut model = Sequential::new();

    model.add(Linear::new(3, 2));
    model.add(Functional::new(Functional::sigmoid()));
    model.add(Linear::new(2, 2));
    model.add(Functional::new(Functional::sigmoid()));

    let mut sgd = SGD::new(model.parameters().unwrap(), SGDOptions::new(0.1));

    let step = |optimizer: &mut SGD, model: Sequential, inputs: Tensor, target: Tensor| {
        // Note: Can't put the following line into closure beacuse
        // zero_grad uses immutable reference and step uses mutable reference.
        optimizer.zero_grad();
        let closure = || {
            let y = model.forward(&[&inputs]);
            let loss = binary_cross_entropy(&y, &target, None, Reduction::Mean);
            backward::backward(&vec![loss.clone()], &vec![], false);
            loss
        };
        optimizer.step(Some(closure))
    };
    let result = step(&mut sgd, model, x.clone(), target);

    //Expected:
    // Result: Tensor: [1.0514469]     size: []
    // Input Grad: Some(Tensor: [0.0003843791, 0.001923646, -0.002772328,
    //                          0.0003843791, 0.001923646, -0.002772328,
    //                          0.0003843791, 0.001923646, -0.002772328,
    //                          0.0003843791, 0.001923646, -0.002772328]   size: [4, 3])
    println!("Result: {:?}", result);
    println!("Input Grad: {:?}", x.grad());
}

#[test]
fn sequential_sgd_two_steps() {
    init_rovo();
    manual_seed(0);
    let x = full(&[4, 3], 1.5, TensorOptions::with_requires_grad());
    let target = ones(&[4, 2], None);
    let mut model = Sequential::new();
    model.add(Linear::new(3, 2));

    let mut sgd = SGD::new(model.parameters().unwrap(), SGDOptions::new(0.1));

    let step = |optimizer: &mut SGD, model: &Sequential, inputs: Tensor, target: &Tensor| {
        let closure = || {
            let y = model.forward(&[&inputs]);
            let loss = target - &y;
            backward::backward(&vec![loss.clone()], &vec![], false);
            loss
        };
        let step = optimizer.step(Some(closure));
        step
    };

    for i in 0..2 {
        let result = step(&mut sgd, &model, x.clone(), &target);

        println!("Result: {:?}", result);
    }
}
