extern crate blas;
extern crate intel_mkl_src;

use blas::*;

fn multiply_nn_naive(m: usize, n: usize, k: usize, alpha: f64,
                        a: &[f64], lda: usize,
                        b: &[f64], ldb: usize, beta: f64,
                        c: &mut [f64], ldc: usize) {
    assert!(m >= lda);
    assert!(m >= ldc);
    assert!(k >= ldb);
    for j in 0..n {
        let b_j = &(b[j*ldb..]);  // j-th column of b 
        for i in 0..m {
            let a_i = &(a[i..]);    //  i-th row of a 
            let mut c_ij = 0.0f64;
            for p in 0..k {
                let a_ip = a_i[p*lda];
                let b_pj = b_j[p];
                c_ij += a_ip * b_pj;
            }
            c[i + j*ldc] = beta * c[i + j*ldc] + alpha * c_ij;
        }
    }
}


fn multiply_nn_blas(m: usize, n: usize, k: usize, alpha: f64,
                             a: &[f64], lda: usize,
                             b: &[f64], ldb: usize, beta: f64,
                             c: &mut [f64], ldc: usize) {
    assert!(m >= lda);
    assert!(m >= ldc);
    assert!(k >= ldb);
    let m : i32 = m.try_into().unwrap();
    let n : i32 = n.try_into().unwrap();
    let k : i32 = k.try_into().unwrap();
    let lda : i32 = lda.try_into().unwrap();
    let ldb : i32 = ldb.try_into().unwrap();
    let ldc : i32 = ldc.try_into().unwrap();
    unsafe {
        dgemm(b'N', b'N', m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}


/// Checks that BLAS works as expected
#[test]
pub fn test_blas() {

    let a = [1.0, 2.0, 3.0,
             1.1, 2.1, 3.1,
             1.2, 2.2, 3.2,
             1.3, 2.3, 3.2f64];
    let lda = 3;

    let b = [1.0, 2.0, 3.0, 4.0,
             1.1, 2.1, 3.1, 4.0f64];
    let ldb=4;

    let ldc = 3;

    let m = 3;
    let n = 2;
    let k = 4;
    let alpha = 1.0f64;
    let beta  = 0.0f64;

    let mut c_ref = [0.0f64 ; 6];
    multiply_nn_naive(m, n, k, alpha, &a, lda, &b, ldb, beta, &mut c_ref, ldc);

    let mut c = [0.0f64 ; 6];
    multiply_nn_blas(m, n, k, alpha, &a, lda, &b, ldb, beta, &mut c, ldc);

    assert_eq!(c, c_ref);
}
