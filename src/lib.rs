#![allow(dead_code)]
#![warn(non_camel_case_types)]
mod aten;
mod autograd;
mod binary_ops;
mod c10;
mod core;
mod engine;
mod ndarry_ext;
mod nn;
mod ops;
mod optim;
mod tensor;
mod util;
mod util_autograd;

fn init_rovo() {
    c10::init();
}
