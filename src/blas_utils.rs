extern crate blas;
extern crate blas_src;

// # Disabling Multithreading in BLAS

// ## MKL

#[cfg(feature="intel-mkl")]
extern "C" {
    fn MKL_Get_Max_Threads() -> i32;
    fn MKL_Set_Num_Threads(num_threads: i32);
}

#[cfg(feature="intel-mkl")]
unsafe fn disable_mt() -> i32 {
    let n_threads = MKL_Get_Max_Threads();
    if n_threads > 1 { MKL_Set_Num_Threads(1); }
    n_threads
}

#[cfg(feature="intel-mkl")]
unsafe fn enable_mt(n_threads: i32) {
    if n_threads > 1 { MKL_Set_Num_Threads(n_threads); }
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

macro_rules! write_gemm {
    ($s:ty, $gemm:ident) => {
        pub fn $gemm(transa: u8, transb: u8,
                    m: usize, n: usize, k: usize, alpha: $s,
                    a: &[$s], lda: usize,
                    b: &[$s], ldb: usize, beta: $s,
                    c: &mut[$s], ldc: usize)
        {
            let lda : i32 = lda.try_into().unwrap();
            let ldb : i32 = ldb.try_into().unwrap();
            let ldc : i32 = ldc.try_into().unwrap();
            let m   : i32 = m.try_into().unwrap();
            let n   : i32 = n.try_into().unwrap();
            let k   : i32 = k.try_into().unwrap();

            unsafe {
                let old = disable_mt();
                blas::$gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
                enable_mt(old);
            }
        }
    }
}

write_gemm!(f32,sgemm);
write_gemm!(f64,dgemm);


macro_rules! write_gemv {
    ($s:ty, $gemv:ident) => {
        pub fn $gemv(trans: u8,
                    m: usize, n: usize, alpha: $s,
                    a: &[$s], lda: usize,
                    x: &[$s], incx: usize, beta: $s,
                    y: &mut[$s], incy: usize)
        {
            let lda : i32 = lda.try_into().unwrap();
            let incx: i32 = incx.try_into().unwrap();
            let incy: i32 = incy.try_into().unwrap();
            let m   : i32 = m.try_into().unwrap();
            let n   : i32 = n.try_into().unwrap();

            unsafe {
                let old = disable_mt();
                blas::$gemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
                enable_mt(old);
            }
        }
    }
}

write_gemv!(f32,sgemv);
write_gemv!(f64,dgemv);

