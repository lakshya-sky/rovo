use crate::aten::{self, native};
use crate::c10::{Scalar, ScalarType};
use crate::tensor::Tensor;
use crate::{AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2, AT_PRIVATE_CASE_TYPE};

// use crate::ndarry_ext;

// blas::sgemm_ assumes that given matrix is in fortran style(COLUMN MAJOR).
// hence we have transpose m1 and m2 accordingly to get corrent output.

pub fn addmm_impl_cpu_(
    result: &Tensor,
    self_: &Tensor,
    mut m1: Tensor,
    mut m2: Tensor,
    beta: Scalar,
    alpha: Scalar,
) {
    dbg!(self_, &m1, &m2);
    assert!(self_.dim() == 2 && m1.dim() == 2 && m2.dim() == 2);
    let _m1 = m1.clone();
    let _m2 = m2.clone();
    let self_sizes = self_.sizes();
    let mut m1_sizes = _m1.sizes();
    let mut m1_strides = _m1.strides();
    let mut m2_sizes = _m2.sizes();
    let mut m2_strides = _m2.strides();

    assert!(
        m1_sizes[1] == m2_sizes[0],
        "mat1 and mat2 can't be multiplied"
    );
    assert!(
        self_sizes[0] == m1_sizes[0] && self_sizes[1] == m2_sizes[1],
        "Input shape is incompatible"
    );
    native::resize(result, self_sizes, None);
    let result_strides = result.strides();
    let result_sizes = result.sizes();

    if result.numel() == 0 {
        return;
    }

    if beta.to::<f64>() != 0.0 && !self_.is_same(result) {
        result.copy(self_, None);
    }

    let transpose_c;
    let c: Tensor;
    // Cast result as matrix a
    if result_strides[0] == 1
        && (result_sizes[1] == 1 || result_strides[1] >= 1usize.max(result_sizes[0]))
    {
        transpose_c = false;
        c = result.clone();
    } else if result_strides[1] == 1
        && (result_sizes[0] == 1 || result_strides[0] >= 1usize.max(result_sizes[1]))
    {
        std::mem::swap(&mut m1, &mut m2);
        std::mem::swap(&mut m1_sizes, &mut m2_sizes);
        std::mem::swap(&mut m1_strides, &mut m2_strides);
        transpose_c = true;
        c = result.clone();
        // todo!()
    } else {
        transpose_c = false;
        // make c FORTRAN contiguous
        let intermediate = result.transpose(0, 1).contiguous();
        intermediate.transpose_(0, 1);
        c = intermediate;
    }
    let m = result_sizes[if transpose_c { 1 } else { 0 }];
    let n = result_sizes[if transpose_c { 0 } else { 1 }];
    let k = m1_sizes[if transpose_c { 0 } else { 1 }];

    // Cast m1 as matrix a
    let transpose_a;
    let a;
    /* Need lda >= max(1, (transpose_a ? k : m)) */
    if m1_strides[if transpose_c { 1 } else { 0 }] == 1
        && m1_strides[if transpose_c { 0 } else { 1 }] >= 1usize.max(m)
    {
        transpose_a = false;
        a = m1.clone();
    } else if m1_strides[if transpose_c { 0 } else { 1 }] == 1
        && m1_strides[if transpose_c { 1 } else { 0 }] >= 1usize.max(k)
    {
        transpose_a = true;
        a = m1.clone();
    } else {
        transpose_a = !transpose_c;
        a = m1.clone();
    }

    // Cast m2 as matrix b
    let transpose_b;
    let b;
    /* Need ldm2_ >= max(1, (transpose_m2 == 'n' ? k : n)) */
    if m2_strides[if transpose_c { 1 } else { 0 }] == 1
        && m2_strides[if transpose_c { 0 } else { 1 }] >= 1usize.max(k)
    {
        transpose_b = false;
        b = m2.clone();
    } else if m2_strides[if transpose_c { 0 } else { 1 }] == 1
        && m2_strides[if transpose_c { 1 } else { 0 }] >= 1usize.max(n)
    {
        transpose_b = true;
        b = m2.clone();
    } else {
        transpose_b = !transpose_c;
        b = m2.clone();
    }
    // dbg!(transpose_c, transpose_a, transpose_b);

    let lda = a.strides()[if transpose_a == transpose_c { 1 } else { 0 }];
    let ldb = b.strides()[if transpose_b == transpose_c { 1 } else { 0 }];
    let ldc = c.strides()[if transpose_c { 0 } else { 1 }];

    // Apply BLAS routine
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2!(result.scalar_type(), "addmm_impl_cpu_", || {
        native::cpublas::gemm(
            if transpose_a {
                native::cpublas::TransposeType::Transpose
            } else {
                native::cpublas::TransposeType::NoTranspose
            },
            if transpose_b {
                native::cpublas::TransposeType::Transpose
            } else {
                native::cpublas::TransposeType::NoTranspose
            },
            m,
            n,
            k,
            alpha.to::<SCALART>(),
            m1.data_ptr().as_ptr(),
            lda,
            m2.data_ptr().as_ptr(),
            ldb,
            beta.to::<SCALART>(),
            c.data_ptr().as_ptr(),
            ldc,
        );
    });
    println!("{:?}", c);
    if !c.is_same(result) {
        result.copy(&c, None);
    }
    dbg!(result);
}

pub fn addmm_cpu_out<'a, S: Into<Scalar>>(
    result: &'a Tensor,
    self_: &Tensor,
    mat1: &Tensor,
    mat2: &Tensor,
    beta: S,
    alpha: S,
) -> &'a Tensor {
    let b_self = aten::expand_size(self_, &[mat1.size(0), mat2.size(1)]);
    {
        addmm_impl_cpu_(
            result,
            &b_self,
            mat1.clone(),
            mat2.clone(),
            beta.into(),
            alpha.into(),
        );
    }
    result
}

pub fn addmm_cpu(
    self_: &Tensor,
    mat1: &Tensor,
    mat2: &Tensor,
    beta: Scalar,
    alpha: Scalar,
) -> Tensor {
    let result = native::empty(&[0], self_.options(), None);
    addmm_cpu_out(&result, self_, mat1, mat2, beta, alpha);
    result
}

pub fn matmul(tensor1: &Tensor, tensor2: &Tensor, consume: bool) -> Tensor {
    let _impl1 = tensor1.get_unsafe_tensor_impl();
    let _impl2 = tensor2.get_unsafe_tensor_impl();
    let dim1 = tensor1.dim();
    let dim2 = tensor2.dim();
    dbg!(tensor1, tensor2);
    let out = match (dim1, dim2) {
        (1, 1) => tensor1.dot(tensor2),
        (2, 2) => tensor1.mm(tensor2, consume),
        (_, _) => todo!(),
    };

    out
}

// pub fn dot(_tensor1: &Tensor, _tensor2: &Tensor) -> Tensor {
//     todo!()
// }

pub fn mm_cpu(mat1: &Tensor, mat2: &Tensor) -> Tensor {
    assert_eq!(mat1.dim(), 2);
    assert_eq!(mat1.dim(), 2);

    let result = super::empty(&[mat1.size(0), mat2.size(1)], mat1.options(), None);
    addmm_cpu_out(&result, &result, mat1, mat2, 0, 1);
    result
}

// pub fn mm_consume(_tensor1: &Tensor, _tensor2: Tensor) -> Tensor {
//     let mat1 = _tensor1._impl.borrow();
//     let mat2 = _tensor2._impl.borrow();

//     assert_eq!(mat1.data.ndim(), 2);
//     assert_eq!(mat2.data.ndim(), 2);

//     let result = ndarry_ext::dot(&mat1.data, &mat2.data);
//     Tensor::from_impl(TensorImpl::new_from_array(result, false))
// }

// pub fn sum(self_: &Tensor, dims: Option<&[usize]>, _keep_dim: bool) -> Tensor {
//     let mut data = self_.get_tensor_impl().data.clone();
//     if let Some(dims) = dims {
//         for dim in dims {
//             data = data.sum_axis(ndarray::Axis(*dim));
//         }
//     } else {
//         data = ndarray::arr0(data.sum()).into_dyn();
//     }
//     Tensor::from_impl(TensorImpl::new_from_array(data, false))
// }

// pub fn mean(self_: &Tensor) -> Tensor {
//     let data = &self_.get_tensor_impl().data;
//     let arr0 = ndarray::arr0(data.mean().unwrap());
//     Tensor::from_impl(TensorImpl::new_from_array(arr0.into_dyn(), false))
// }
