use super::*;
use crate::cuda;
use cuda::DevPtr;
use std::os::raw::c_void;

#[allow(non_camel_case_types)]
pub type cublasOperation_t = ::std::os::raw::c_uint;

pub const CUBLAS_OP_N: cublasOperation_t = 0;
pub const CUBLAS_OP_T: cublasOperation_t = 1;
pub const CUBLAS_OP_C: cublasOperation_t = 2;
pub const CUBLAS_OP_HERMITAN: cublasOperation_t = 2;
pub const CUBLAS_OP_CONJG: cublasOperation_t = 3;

extern "C" {
    pub fn cublasSgemv_v2(
        handle: cublasHandle_t,
        trans: cublasOperation_t,
        m: i32,
        n: i32,
        alpha: *const f32,
        A: *const c_void,
        lda: i32,
        x: *const c_void,
        incx: i32,
        beta: *const f32,
        y: *mut c_void,
        incy: i32,
    ) -> cublasStatus_t;

    pub fn cublasDgemv_v2(
        handle: cublasHandle_t,
        trans: cublasOperation_t,
        m: i32,
        n: i32,
        alpha: *const f64,
        A: *const c_void,
        lda: i32,
        x: *const f64,
        incx: i32,
        beta: *const c_void,
        y: *mut c_void,
        incy: i32,
    ) -> cublasStatus_t;

    pub fn cublasSgemm_v2(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f32,
        A: *const c_void,
        lda: i32,
        B: *const c_void,
        ldb: i32,
        beta: *const f32,
        C: *mut c_void,
        ldc: i32,
    ) -> cublasStatus_t;

    pub fn cublasDgemm_v2(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f64,
        A: *const c_void,
        lda: i32,
        B: *const c_void,
        ldb: i32,
        beta: *const f64,
        C: *mut c_void,
        ldc: i32,
    ) -> cublasStatus_t;

    pub fn cublasSgeam(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i32,
        n: i32,
        alpha: *const f32,
        A: *const c_void,
        lda: i32,
        beta: *const f32,
        B: *const c_void,
        ldb: i32,
        C: *mut c_void,
        ldc: i32,
    ) -> cublasStatus_t;

    pub fn cublasDgeam(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i32,
        n: i32,
        alpha: *const f64,
        A: *const c_void,
        lda: i32,
        beta: *const f64,
        B: *const c_void,
        ldb: i32,
        C: *mut c_void,
        ldc: i32,
    ) -> cublasStatus_t;

}

pub fn dgemm (handle: &Context,
             transa: u8,
             transb: u8,
             m: usize,
             n: usize,
             k: usize,
             alpha: f64,
             a: &DevPtr<f64>,
             lda: usize,
             b: &DevPtr<f64>,
             ldb: usize,
             beta: f64,
             c: &mut DevPtr<f64>,
             ldc: usize
            ) -> Result<(), CublasError>
{
    let transa = match transa {
            b'N' | b'n' => CUBLAS_OP_N,
            b'T' | b't' => CUBLAS_OP_T,
            _ => panic!("N or T expected")
            //_ => return Err(CublasError(INVALID_VALUE))
    };

    let transb = match transb {
            b'N' | b'n' => CUBLAS_OP_N,
            b'T' | b't' => CUBLAS_OP_T,
            _ => panic!("N or T expected")
            //_ => return Err(CublasError(INVALID_VALUE))
    };

    let status = unsafe {
        cublasDgemm_v2(
            handle.as_cublasHandle_t(),
            transa,
            transb,
            m as i32,
            n as i32,
            k as i32,
            &alpha as *const f64,
            a.as_raw_ptr() as *const c_void,
            lda as i32,
            b.as_raw_ptr() as *const c_void,
            ldb as i32,
            &beta as *const f64,
            c.as_raw_mut_ptr() as *mut c_void,
            ldc as i32,
        )
    };

    wrap_error( (), status)
}


pub fn sgemm (handle: &Context,
             transa: u8,
             transb: u8,
             m: usize,
             n: usize,
             k: usize,
             alpha: f32,
             a: &DevPtr<f32>,
             lda: usize,
             b: &DevPtr<f32>,
             ldb: usize,
             beta: f32,
             c: &mut DevPtr<f32>,
             ldc: usize
            ) -> Result<(), CublasError>
{
    let transa = match transa {
            b'N' | b'n' => CUBLAS_OP_N,
            b'T' | b't' => CUBLAS_OP_T,
            _ => panic!("N or T expected")
            //_ => return Err(CublasError(INVALID_VALUE))
    };

    let transb = match transb {
            b'N' | b'n' => CUBLAS_OP_N,
            b'T' | b't' => CUBLAS_OP_T,
            _ => panic!("N or T expected")
            //_ => return Err(CublasError(INVALID_VALUE))
    };

    let status = unsafe {
        cublasSgemm_v2(
            handle.as_cublasHandle_t(),
            transa,
            transb,
            m as i32,
            n as i32,
            k as i32,
            &alpha as *const f32,
            a.as_raw_ptr() as *const c_void,
            lda as i32,
            b.as_raw_ptr() as *const c_void,
            ldb as i32,
            &beta as *const f32,
            c.as_raw_mut_ptr() as *mut c_void,
            ldc as i32)
    };

    wrap_error( (), status)
}


pub fn dgeam (handle: &Context,
             transa: u8,
             transb: u8,
             m: usize,
             n: usize,
             alpha: f64,
             a: &DevPtr<f64>,
             lda: usize,
             beta: f64,
             b: &DevPtr<f64>,
             ldb: usize,
             c: &mut DevPtr<f64>,
             ldc: usize
            ) -> Result<(), CublasError>
{
    let transa = match transa {
            b'N' | b'n' => CUBLAS_OP_N,
            b'T' | b't' => CUBLAS_OP_T,
            _ => panic!("N or T expected")
            //_ => return Err(CublasError(INVALID_VALUE))
    };

    let transb = match transb {
            b'N' | b'n' => CUBLAS_OP_N,
            b'T' | b't' => CUBLAS_OP_T,
            _ => panic!("N or T expected")
            //_ => return Err(CublasError(INVALID_VALUE))
    };

    let status = unsafe {
        cublasDgeam(
            handle.as_cublasHandle_t(),
            transa,
            transb,
            m as i32,
            n as i32,
            &alpha as *const f64,
            a.as_raw_ptr() as *const c_void,
            lda as i32,
            &beta as *const f64,
            b.as_raw_ptr() as *const c_void,
            ldb as i32,
            c.as_raw_mut_ptr() as *mut c_void,
            ldc as i32)
    };

    wrap_error( (), status)
}

pub fn sgeam (handle: &Context,
             transa: u8,
             transb: u8,
             m: usize,
             n: usize,
             alpha: f32,
             a: &DevPtr<f32>,
             lda: usize,
             beta: f32,
             b: &DevPtr<f32>,
             ldb: usize,
             c: &mut DevPtr<f32>,
             ldc: usize
            ) -> Result<(), CublasError>
{
    let transa = match transa {
            b'N' | b'n' => CUBLAS_OP_N,
            b'T' | b't' => CUBLAS_OP_T,
            _ => panic!("N or T expected")
            //_ => return Err(CublasError(INVALID_VALUE))
    };

    let transb = match transb {
            b'N' | b'n' => CUBLAS_OP_N,
            b'T' | b't' => CUBLAS_OP_T,
            _ => panic!("N or T expected")
            //_ => return Err(CublasError(INVALID_VALUE))
    };

    let status = unsafe {
        cublasSgeam(
            handle.as_cublasHandle_t(),
            transa,
            transb,
            m as i32,
            n as i32,
            &alpha as *const f32,
            a.as_raw_ptr() as *const c_void,
            lda as i32,
            &beta as *const f32,
            b.as_raw_ptr() as *const c_void,
            ldb as i32,
            c.as_raw_mut_ptr() as *mut c_void,
            ldc as i32)
    };

    wrap_error( (), status)
}

