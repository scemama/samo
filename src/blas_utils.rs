extern crate blas;
extern crate blas_src;

// # Disabling Multithreading in BLAS

// ## MKL

#[cfg(feature="intel-mkl")]
extern "C" {
    fn omp_get_num_threads() -> i32;
    fn omp_set_num_threads(num_threads: i32);
}

#[cfg(feature="intel-mkl")]
unsafe fn disable_mt() -> i32 {
    let n_threads = omp_get_num_threads();
    omp_set_num_threads(1);
    n_threads
}

#[cfg(feature="intel-mkl")]
unsafe fn enable_mt(n_threads: i32) {
    omp_set_num_threads(n_threads);
}


// ## BLIS

#[cfg(feature="blis")]
extern "C" {
    fn bli_thread_get_num_threads() -> i32;
    fn bli_thread_set_num_threads(num_threads: i32);
}

#[cfg(feature="blis")]
unsafe fn disable_mt() -> i32 {
    let n_threads = bli_thread_get_num_threads();
    bli_thread_set_num_threads(1);
    n_threads
}

#[cfg(feature="blis")]
unsafe fn enable_mt(n_threads: i32) {
    bli_thread_set_num_threads(n_threads);
}

// ## OpenBLAS

#[cfg(feature="openblas")]
extern "C" {
    fn openblas_get_num_threads() -> i32;
    fn openblas_set_num_threads(num_threads: i32);
}

#[cfg(feature="openblas")]
unsafe fn disable_mt() -> i32 {
    let n_threads = openblas_get_num_threads();
    openblas_set_num_threads(1);
    n_threads
}

#[cfg(feature="openblas")]
unsafe fn enable_mt(n_threads: i32) {
    openblas_set_num_threads(n_threads);
}


// ## Netlib

#[cfg(feature="netlib")]
unsafe fn disable_mt() -> i32 {
    0
}

#[cfg(feature="netlib")]
unsafe fn enable_mt(_: i32) {
}

// ## Accelerate

#[cfg(feature="accelerate")]
unsafe fn disable_mt() -> i32 {
    match std::env::var("VECLIB_MAXIMUM_THREADS") {
        Ok(val) => val.parse::<i32>().unwrap_or(1),
        Err(_) => 1,
    }
}

#[cfg(feature="accelerate")]
unsafe fn enable_mt(n_threads: i32) {
    std::env::set_var("VECLIB_MAXIMUM_THREADS",n_threads.to_string());
}




/// # BLAS Interfaces
/// A constant representing the leading dimension of arrays in tiles,
/// which is also the maximum number of rows and columns a `Tile` can
/// have.
/// BLAS operations
///
pub trait Float: num::traits::Float + Sync + Send {
    fn blas_gemm(transa: u8, transb: u8,
            m: usize, n: usize, k: usize, alpha: Self,
            a: &[Self], lda: usize,
            b: &[Self], ldb: usize, beta: Self,
            c: &mut[Self], ldc: usize);
}


impl Float for f64 {
    /// BLAS DGEMM
    fn blas_gemm(transa: u8, transb: u8,
                m: usize, n: usize, k: usize, alpha: Self,
                a: &[Self], lda: usize,
                b: &[Self], ldb: usize, beta: Self,
                c: &mut[Self], ldc: usize)
    {
        let lda : i32 = lda.try_into().unwrap();
        let ldb : i32 = ldb.try_into().unwrap();
        let ldc : i32 = ldc.try_into().unwrap();
        let m   : i32 = m.try_into().unwrap();
        let n   : i32 = n.try_into().unwrap();
        let k   : i32 = k.try_into().unwrap();

        unsafe {
              let old = disable_mt();
              blas::dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
              enable_mt(old);
        }
    }
}

impl Float for f32 {
    /// BLAS SGEMM
    fn blas_gemm(transa: u8, transb: u8,
                m: usize, n: usize, k: usize, alpha: Self,
                a: &[Self], lda: usize,
                b: &[Self], ldb: usize, beta: Self,
                c: &mut[Self], ldc: usize)
    {
        let lda : i32 = lda.try_into().unwrap();
        let ldb : i32 = ldb.try_into().unwrap();
        let ldc : i32 = ldc.try_into().unwrap();
        let m   : i32 = m.try_into().unwrap();
        let n   : i32 = n.try_into().unwrap();
        let k   : i32 = k.try_into().unwrap();

        unsafe {
              let old = disable_mt();
              blas::sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
              enable_mt(old);
        }

    }
}

