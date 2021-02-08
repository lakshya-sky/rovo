use rovo::{
    aten::native::wrapped_scalar_tensor,
    autograd::tensor,
    init_rovo,
    nn::{log_softmax, nll_loss, NLLLossFuncOptions},
    tensor::loss::Reduction,
};

#[test]
fn nllloss() {
    init_rovo();
    let input = tensor(
        &[
            -0.1315, -3.1315, -2.5315, -3.7038, -0.1038, -2.6038, -2.3422, -1.3422, -0.4422,
        ],
        None,
    );
    input.resize(&[3, 3], None);
    let target = tensor(&[1i64, 0, 2], None);
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

#[test]
fn test_log_softmax() {
    init_rovo();
    let input = tensor(&[0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], None);
    input.resize(&[2, 5], None);
    let output = log_softmax(&input, 1);
    println!("Output: {:?}", output);
    //Expected
    /*
    (float) [0] = -4.45191431
    (float) [1] = -3.45191431
    (float) [2] = -2.45191431
    (float) [3] = -1.45191443
    (float) [4] = -0.4519144
    (float) [5] = -4.45191431
    (float) [6] = -3.45191431
    (float) [7] = -2.45191431
    (float) [8] = -1.45191443
    (float) [9] = -0.4519144

    */
}

// #[test]
// fn nll_loss_test() {
//     let input = tensor(
//         &[
//             -0.1315, -3.1315, -2.5315, -3.7038, -0.1038, -2.6038, -2.3422, -1.3422, -0.4422,
//         ],
//         None,
//     );

// }
