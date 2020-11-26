use num::{cast, NumCast};
use std::ffi::c_void;
#[repr(i8)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub enum TransposeType {
    Transpose = 84,   // ASCII 'T'
    NoTranspose = 78, //ASCII 'N'
}
pub fn normalize_last_dims(
    transa: TransposeType,
    transb: TransposeType,
    m: usize,
    n: usize,
    k: usize,
    lda: &mut usize,
    ldb: &mut usize,
    ldc: &mut usize,
) {
    if n == 1 {
        *ldc = m;
    }

    if transa != TransposeType::NoTranspose {
        if m == 1 {
            *lda = k;
        }
    } else if k == 1 {
        *lda = m;
    }

    if transb != TransposeType::NoTranspose {
        if k == 1 {
            *ldb = n;
        }
    } else if n == 1 {
        *ldb = k;
    }
}

pub fn gemm<T: NumCast>(
    trans_a: TransposeType,
    trans_b: TransposeType,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: *const c_void,
    mut lda: usize,
    b: *const c_void,
    mut ldb: usize,
    beta: T,
    c: *mut c_void,
    mut ldc: usize,
) {
    normalize_last_dims(trans_a, trans_b, m, n, k, &mut lda, &mut ldb, &mut ldc);
    match std::any::type_name::<T>() {
        "f32" => gemm_f32(
            trans_a,
            trans_b,
            m,
            n,
            k,
            cast(alpha).unwrap(),
            a as *const f32,
            cast(lda).unwrap(),
            b as *const f32,
            cast(ldb).unwrap(),
            cast(beta).unwrap(),
            c as *mut f32,
            cast(ldc).unwrap(),
        ),
        _ => todo!(),
    }
}

pub fn gemm_f32(
    trans_a: TransposeType,
    trans_b: TransposeType,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    mut lda: usize,
    b: *const f32,
    mut ldb: usize,
    beta: f32,
    c: *mut f32,
    mut ldc: usize,
) {
    normalize_last_dims(trans_a, trans_b, m, n, k, &mut lda, &mut ldb, &mut ldc);
    unsafe {
        blas_sys::sgemm_(
            &(trans_a as i8),
            &(trans_b as i8),
            &(m as i32),
            &(n as i32),
            &(k as i32),
            &alpha,
            a,
            &(lda as i32),
            b,
            &(ldb as i32),
            &beta,
            c,
            &(ldc as i32),
        );
    }
}
