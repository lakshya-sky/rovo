# Rovo

## Experimental Tensor Libary in Rust inspired from Pytorch.

To-do:

- [x] Use not_ready for evaluation of function.
- [x] Impl -, /, and unary -.
- [x] Basic Linear Layer.
- [X] function for comparing two tensors. (shallow_comapre that compares only data and shape)
- [X] Random number Generator. (for CPU only)
- [ ] basic Optimizer 
    - [X] SGD
    - [ ] Adam
- [-] TensorIterator and TensorInteratorConfig.
- [ ] Binary_Cross_Entropy.
- [ ] Impl OrderedDict similiar to python's.
- [ ] Pow, Exp operators.
- [ ] Improve Debug trait impls.
- [ ] Improve sum() & view() methods.
- [ ] Use queue for evaluate_outputs.


Corner-cases:

- [ ] RTTI for module name inference.

Notes:

- To run tests run `cargo test -- --test-threads=1`. This will make sure that tests are executing on single threads. Because running tests in parallel makes some tests to fail.
