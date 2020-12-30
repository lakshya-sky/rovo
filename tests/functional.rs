use rovo::{
    aten::native::wrapped_scalar_tensor,
    autograd::tensor,
    nn::{nll_loss, NLLLossFuncOptions},
    tensor::loss::Reduction,
};

#[test]
fn nllloss() {
    let input = tensor(
        &[
            -0.1315, -3.1315, -2.5315, -3.7038, -0.1038, -2.6038, -2.3422, -1.3422, -0.4422,
        ],
        None,
    );
    let target = tensor(&[1.0f32, 2.0, 3.0], None);
    let output = nll_loss(
        &input,
        &target,
        NLLLossFuncOptions::new()
            .set_ignore_index(-100)
            .set_reduction(Reduction::Mean),
    );
    let expected = wrapped_scalar_tensor(2.4258f32.into());
    println!("Expected: {:?}, Output: {:?}", expected, output);
}
