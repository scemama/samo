pub fn blas_dgemm(transa: u8, transb: u8,
              m: usize, n:usize, k:usize, _alpha: f64,
              a: &[f64], lda: usize,
              b: &[f64], ldb: usize, _beta: f64,
              c: &mut[f64], ldc: usize) {
    let m : i32 = m.try_into().unwrap();
    let n : i32 = n.try_into().unwrap();
    let k : i32 = k.try_into().unwrap();
    let lda : i32 = lda.try_into().unwrap();
    let ldb : i32 = ldb.try_into().unwrap();
    let ldc : i32 = ldc.try_into().unwrap();
    unsafe {
        blas::dgemm(transa, transb, m, n, k, _alpha, a, lda, b, ldb, _beta, c, ldc);
    }

}

pub fn blas_sgemm(transa: u8, transb: u8,
              m: usize, n:usize, k:usize, _alpha: f32,
              a: &[f32], lda: usize,
              b: &[f32], ldb: usize, _beta: f32,
              c: &mut[f32], ldc: usize) {
    let m : i32 = m.try_into().unwrap();
    let n : i32 = n.try_into().unwrap();
    let k : i32 = k.try_into().unwrap();
    let lda : i32 = lda.try_into().unwrap();
    let ldb : i32 = ldb.try_into().unwrap();
    let ldc : i32 = ldc.try_into().unwrap();
    unsafe {
        blas::sgemm(transa, transb, m, n, k, _alpha, a, lda, b, ldb, _beta, c, ldc);
    }

}


