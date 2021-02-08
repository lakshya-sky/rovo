# Rovo

## Experimental Tensor Libary in Rust inspired from Pytorch.

Last:

- softmax_kernel.rs:90 code is commented due to malloc error.
- loss_nll should use loss_nll_forward with gradient support.
- Handle case where iterator has no output. loops.rs:execute_op

To-do:

- [x] Empty tensor creation.
- [x] Read Tensor by index and print Tensor.
  - Index trait for [] enforces to return reference while we need value, hence I am using Get trait to get tensor content using get() method.
- [ ] Make distributions consistent with Pytorch.
- [ ] Check for backprop and make it consistent with Pytorch.

Notes:

- To run tests run `cargo test -- --test-threads=1`. This will make sure that tests are executing on single threads. Because running tests in parallel makes some tests to fail.
