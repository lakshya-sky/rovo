use crate::core::NoGradGuard;
use crate::tensor::*;

struct OptimizerOptions {}
struct OptimizerParamGroup {
    params: Vec<Tensor>,
    // options: OptimizerOptions
}

impl OptimizerParamGroup {
    pub fn new(params: Vec<Tensor>) -> Self {
        Self { params }
    }

    pub fn params(&self) -> &Vec<Tensor> {
        &self.params
    }
}

trait Optimizer {
    fn step(&mut self) -> Option<Tensor>;
    fn param_groups(&self) -> &Vec<OptimizerParamGroup>;
    fn zero_grad(&self) {
        for group in self.param_groups() {
            for p in group.params() {
                if let Some(grad) = p.grad().as_mut() {
                    grad.detach_();
                    grad.zero_();
                }
            }
        }
    }
}
pub struct SGD {
    param_groups: Vec<OptimizerParamGroup>,
}

impl SGD {
    pub fn new(params: Vec<Tensor>) -> Self {
        SGD::new_from_param_group(vec![OptimizerParamGroup::new(params)])
    }

    fn new_from_param_group(param_groups: Vec<OptimizerParamGroup>) -> Self {
        Self { param_groups }
    }
}

impl Optimizer for SGD {
    fn step(&mut self) -> Option<Tensor> {
        let _guard = NoGradGuard::default();
        let weight_decay = 0.0;
        let learning_rate = 0.001;
        for group in &mut self.param_groups {
            for p in group.params() {
                match p.grad().as_mut() {
                    Some(d_p_) => {
                        if weight_decay != 0.0 {
                            // eprintln!("Weight Grad Before: {:?}", borrow_);
                            let d_p = &d_p_.clone() + weight_decay;
                            d_p_.move_tensor(d_p);
                        }
                        let d_p = d_p_.clone();
                        p.add_(&d_p, -1.0 * learning_rate);
                    }
                    None => continue,
                }
            }
        }
        None
    }

    fn param_groups(&self) -> &Vec<OptimizerParamGroup> {
        self.param_groups.as_ref()
    }
}

// #[cfg(test)]
// mod test {
//     use super::{Optimizer, SGD};
//     use crate::autograd::backward;
//     use crate::nn::Module;
//     use crate::nn::{Functional, Linear};
//     use crate::tensor::Tensor;

//     #[test]
//     fn test_sgd_step() {
//         let linear = Linear::new(2, 1);
//         let sigmoid = Functional::new(Functional::sigmoid());
//         let mut sgd = SGD::new(linear.parameters());
//         let x = Tensor::from_scalar(&[2, 2], 2.0, true);
//         sgd.zero_grad();
//         let h = linear.forward(&[&x]);
//         let y = sigmoid.forward(&[&h]);
//         let target = Tensor::ones(&[2, 1]);
//         let result = crate::tensor::binary_cross_entropy(
//             &y,
//             &target,
//             None,
//             crate::tensor::loss::Reduction::Mean,
//         );
//         println!("Y: {:?}", y);
//         println!("Result: {:?}", result);

//         backward::backward(&vec![result], &vec![], false);
//         {
//             println!("Weights Before step: {:?}", linear.parameters());
//         }
//         sgd.step();
//         println!("Weights after step: {:?}", linear.parameters());
//     }
// }
