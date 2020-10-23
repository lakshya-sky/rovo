# Rovo

## Experimental Tensor Libary in Rust inspired from Pytorch.

Last:

- Insted of type UniqueVoidPointer use struct with explicit type as c_void.

To-do:

- [x] Use not_ready for evaluation of function.
- [x] Impl -, /, and unary -.
- [x] Basic Linear Layer.
- [x] function for comparing two tensors. (shallow_comapre that compares only data and shape)
- [x] Random number Generator. (for CPU only)
- [x] Allow seed with manual_seed.
- [ ] basic Optimizer
  - [x] SGD
  - [ ] Adam
- [-] TensorIterator and TensorInteratorConfig.
- [ ] overload [] operator for Tensor. https://doc.rust-lang.org/src/alloc/vec.rs.html#1970-1977
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
