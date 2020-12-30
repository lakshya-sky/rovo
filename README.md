# Rovo

## Experimental Tensor Libary in Rust inspired from Pytorch.

Sample Code:

<<<<<<< HEAD
Following code snippet is for matrix multiplication of two tensors and their gradients in the backward pass (which is calculated using tensor of ones as an input)

```
use rovo::{autograd, tensor::log_softmax, c10::TensorOptions, init_rovo};

fn main() {
    // Initialize Allocator and CPUGenerators on which Tensor Allocation
    // and random number generation work.
    rovo::init_rovo();
    let x = autograd::full(&[2, 2], 3.0, TensorOptions::with_requires_grad());
    let w = autograd::full(&[2, 1], 2.0, TensorOptions::with_requires_grad());
    let result = x.mm(&w, true);
    println!("Result: {:?}", result);
    autograd::backward(&vec![result], &vec![], false);
    println!("{:?}", x.grad().unwrap().as_ref());
    println!("{:?}", w.grad().unwrap().as_ref());
}


---- Prints ----
Result: Tensor: [12.0, 12.0]    size: [2, 1]
gradX:  Tensor: [2.0, 2.0, 2.0, 2.0]    size: [2, 2]
gradW:  Tensor: [6.0, 6.0]      size: [2, 1]
```

following is operation for a single layer of full connected layer ((Wx + b) -> sigmoid -> bceloss) and
getting the input's gradient.

```
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
---- Prints ----
Loss: Tensor: [1.02007] size: []
Input Grad: Some(Tensor: [0.019757522, -0.058482096, 0.06944271, 0.027283773, 0.019757522, -0.058482096, 0.06944271, 0.027283773]       size: [2, 4])
```
=======
```
use rovo::{autograd, tensor::log_softmax, c10::TensorOptions, init_rovo};

fn main() {
    // Initialize Allocator and CPUGenerators on which Tensor Allocation 
    // and random number generation work.
    rovo::init_rovo();
    let x = autograd::full(&[2, 2], 3.0, TensorOptions::with_requires_grad());
    let w = autograd::full(&[2, 1], 2.0, TensorOptions::with_requires_grad());
    let result = x.mm(&w, true);
    println!("Result: {:?}", result);
    autograd::backward(&vec![result], &vec![], false);
    println!("{:?}", x.grad().unwrap().as_ref());
    println!("{:?}", w.grad().unwrap().as_ref());
}

 
---- Prints ----
Result: Tensor: [12.0, 12.0]    size: [2, 1] 
gradX:  Tensor: [2.0, 2.0, 2.0, 2.0]    size: [2, 2]
gradW:  Tensor: [6.0, 6.0]      size: [2, 1]
```
```
fn linear_backward_test() {
    init_rovo();
    manual_seed(0);
    let linear = Linear::new(4, 3);
    let x = autograd::full(&[2, 4], 3.0, TensorOptions::with_requires_grad());
    let y = linear.forward(&[&x]);
    println!("Result: {:?}", y);
    println!("Mean: {:?}", y.mean());
    backward::backward(&vec![y], &vec![], false);
    println!("Input Grad: {:?}", x.grad());
}
---- Prints ----
Result: Tensor: [-2.0227153, 0.6529779, -0.6904765, -2.0227153, 0.6529779, -0.6904765]  size: [2, 3]
Mean: Tensor: [-0.68673795]     size: []
Input Grad: Some(Tensor: [-0.04011544, 0.08910112, -0.09542262, -0.011634579, -0.04011544, 0.08910112, -0.09542262, -0.011634579]       size: [2, 4])

```
To-do:

- [x] Empty tensor creation.
- [x] Read Tensor by index and print Tensor.
  - Index trait for [] enforces to return reference while we need value, hence I am using Get trait that has get() method.
- [x] Make distributions consistent with Pytorch.
- [x] Check for backprop and make it consistent with Pytorch.
>>>>>>> Update README

Notes:

- To run tests run `cargo test -- --test-threads=1`. This will make sure that tests are executing on single threads. Parellel tests are using shaered variables which will make some tests fail.
