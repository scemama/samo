#![allow(non_camel_case_types)]
///! This module is a minimal interface to CuBLAS functions

use crate::cuda;
use crate::cuda::{Stream, cudaStream_t};
use std::os::raw::c_void;

use cuda::DevPtr;


//  # Error handling
//  # --------------

use std::{fmt, error};

pub struct CublasError(::std::os::raw::c_uint);

use ::std::os::raw::c_uint as cublasStatus_t;

const SUCCESS: cublasStatus_t = 0;
const NOT_INITIALIZED: cublasStatus_t = 1;
const ALLOC_FAILED: cublasStatus_t = 3;
const INVALID_VALUE: cublasStatus_t = 7;
const ARCH_MISMATCH: cublasStatus_t = 8;
const MAPPING_ERROR: cublasStatus_t = 11;
const EXECUTION_FAILED: cublasStatus_t = 13;
const INTERNAL_ERROR: cublasStatus_t = 14;
const NOT_SUPPORTED: cublasStatus_t = 15;
const LICENSE_ERROR: cublasStatus_t = 16;

fn fmt_error(s: &CublasError, f: &mut fmt::Formatter<'_>) -> fmt::Result {
       match s.0 {
        SUCCESS           =>  write!(f, "Success"),
        NOT_INITIALIZED   =>  write!(f, "Not initialized"),
        ALLOC_FAILED      =>  write!(f, "Allocation failed"),
        INVALID_VALUE     =>  write!(f, "Invalid value"),
        ARCH_MISMATCH     =>  write!(f, "Arch mismatch"),
        MAPPING_ERROR     =>  write!(f, "Mapping error"),
        EXECUTION_FAILED  =>  write!(f, "Execution failed"),
        INTERNAL_ERROR    =>  write!(f, "Interal error"),
        NOT_SUPPORTED     =>  write!(f, "Not supported"),
        LICENSE_ERROR     =>  write!(f, "License error"),
        i  => write!(f, "Error {i}"),
       }
}

impl fmt::Display for CublasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_error(self,f)
    }
}

impl fmt::Debug for CublasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_error(self,f)
    }
}

impl error::Error for CublasError {}

fn wrap_error<T>(output: T, e: cublasStatus_t) -> Result<T, CublasError> {
    match e {
        0 => Ok(output),
        _ => Err(CublasError(e)),
    }
}


//  # CuBLAS context
//  # --------------

type cublasHandle_t = *mut c_void;

#[cfg(feature = "cublas")]
#[link(name = "cublas")]
#[link(name = "cublasLt")]
extern "C" {}

extern "C" {
    pub fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t;
    pub fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t;
}

use std::sync::{Arc};
use std::ptr::NonNull;

#[derive(Debug)]
pub struct CublasContext {
    handle: NonNull<c_void>,
}

impl CublasContext {

    fn new() -> Result<Self, CublasError> {
        let mut handle = std::ptr::null_mut();
        let rc = unsafe { cublasCreate_v2(&mut handle as *mut *mut c_void) };
        NonNull::new(handle).map(|handle| Self { handle })
            .ok_or(CublasError(rc))
    }

    fn as_raw_mut_ptr(&self) -> *mut c_void {
        self.handle.as_ptr()
    }

    pub fn as_raw_ptr(&self) -> *const c_void {
        self.handle.as_ptr()
    }
}

impl Drop for CublasContext {
    fn drop(&mut self) {
        unsafe { cublasDestroy_v2(self.as_raw_mut_ptr() as *mut c_void) };
    }
}

#[derive(Debug, Clone)]
pub struct Context(Arc<CublasContext>);

impl Context {

    pub fn new() -> Result<Self, CublasError> {
        CublasContext::new().map(|context| Self(Arc::new(context)))
    }

    pub fn as_cublasHandle_t(&self) -> cublasHandle_t {
        self.0.as_raw_mut_ptr()
    }

    pub fn as_raw_ptr(&self) -> *const c_void {
        self.0.as_raw_ptr()
    }
}




//  # Memory operations
//  # -----------------

extern "C" {
    pub fn cublasSetMatrix(
        rows: i32,
        cols: i32,
        elemSize: i32,
        A: *const c_void,
        lda: i32,
        B: *mut c_void,
        ldb: i32,
    ) -> cublasStatus_t;

    pub fn cublasSetMatrixAsync(
        rows: i32,
        cols: i32,
        elemSize: i32,
        A: *const c_void,
        lda: i32,
        B: *mut c_void,
        ldb: i32,
        stream: cudaStream_t,
    ) -> cublasStatus_t;

    pub fn cublasScopy_v2(
        handle: cublasHandle_t,
        n: i32,
        x: *const f32,
        incx: i32,
        y: *mut f32,
        incy: i32,
    ) -> cublasStatus_t;

    pub fn cublasGetMatrix(
        rows: i32,
        cols: i32,
        elemSize: i32,
        A: *const c_void,
        lda: i32,
        B: *mut c_void,
        ldb: i32,
    ) -> cublasStatus_t;

    pub fn cublasGetMatrixAsync(
        rows: i32,
        cols: i32,
        elemSize: i32,
        A: *const c_void,
        lda: i32,
        B: *mut c_void,
        ldb: i32,
        stream: cudaStream_t,
    ) -> cublasStatus_t;
}


/// Sends an array to the device
pub fn set_matrix<T>(rows: usize, cols: usize, a: &[T], lda: usize, d_a: &mut DevPtr<T>, d_lda: usize) -> Result<(), CublasError> {
    // Check that the slice is not empty and the dimensions are valid.
    if a.is_empty() || rows == 0 || cols == 0 {
        return Err(CublasError(INVALID_VALUE));
    }

    // Check that the slice length is at least as large as rows * lda to ensure memory safety.
    if a.len() < cols * lda {
        return Err(CublasError(INVALID_VALUE));
    }

    // Calculate the size of the elements in the slice.
    let elem_size = std::mem::size_of::<T>() as i32;

    // Perform the FFI call.
    let status = unsafe {
        cublasSetMatrix(
            rows as i32,
            cols as i32,
            elem_size,
            a.as_ptr() as *const c_void,
            lda as i32,
            d_a.as_raw_mut_ptr(),
            d_lda as i32,
        )
    };

    wrap_error( (), status)
}

/// Sends an array to the device
pub fn set_matrix_async<T>(rows: usize, cols: usize, a: &[T], lda: usize, d_a: &mut DevPtr<T>, d_lda: usize, stream: &Stream) -> Result<(), CublasError> {
    // Check that the slice is not empty and the dimensions are valid.
    if a.is_empty() || rows == 0 || cols == 0 {
        return Err(CublasError(INVALID_VALUE));
    }

    // Check that the slice length is at least as large as rows * lda to ensure memory safety.
    if a.len() < cols * lda {
        return Err(CublasError(INVALID_VALUE));
    }

    // Calculate the size of the elements in the slice.
    let elem_size = std::mem::size_of::<T>() as i32;

    // Perform the FFI call.
    let status = unsafe {
        cublasSetMatrixAsync(
            rows as i32,
            cols as i32,
            elem_size,
            a.as_ptr() as *const c_void,
            lda as i32,
            d_a.as_raw_mut_ptr(),
            d_lda as i32,
            stream.as_cudaStream_t(),
        )
    };

    wrap_error( (), status)
}


/// Fetches an array from the device
pub fn get_matrix<T>(rows: usize, cols: usize, d_a: &DevPtr<T>, d_lda: usize, a: &mut [T], lda: usize) -> Result<(), CublasError> {
    // Check that the slice is not empty and the dimensions are valid.
    if a.is_empty() || rows == 0 || cols == 0 {
        return Err(CublasError(INVALID_VALUE));
    }

    // Check that the slice length is at least as large as rows * lda to ensure memory safety.
    if a.len() < cols * lda {
        return Err(CublasError(INVALID_VALUE));
    }

    // Calculate the size of the elements in the slice.
    let elem_size = std::mem::size_of::<T>() as i32;

    let ptr_a = d_a.as_raw_ptr();

    // Perform the FFI call.
    let status = unsafe {
        cublasGetMatrix(
            rows as i32,
            cols as i32,
            elem_size,
            ptr_a,
            d_lda as i32,
            a.as_mut_ptr() as *mut c_void,
            lda as i32,
        )
    };

    wrap_error( (), status)
}

/// Fetches an array from the device
pub fn get_matrix_async<T>(rows: usize, cols: usize, d_a: &DevPtr<T>, d_lda: usize, a: &mut [T], lda: usize, stream: &Stream) -> Result<(), CublasError> {
    // Check that the slice is not empty and the dimensions are valid.
    if a.is_empty() || rows == 0 || cols == 0 {
        return Err(CublasError(INVALID_VALUE));
    }

    // Check that the slice length is at least as large as rows * lda to ensure memory safety.
    if a.len() < cols * lda {
        return Err(CublasError(INVALID_VALUE));
    }

    // Calculate the size of the elements in the slice.
    let elem_size = std::mem::size_of::<T>() as i32;

    let ptr_a = d_a.as_raw_ptr();

    // Perform the FFI call.
    let status = unsafe {
        cublasGetMatrixAsync(
            rows as i32,
            cols as i32,
            elem_size,
            ptr_a,
            d_lda as i32,
            a.as_mut_ptr() as *mut c_void,
            lda as i32,
            stream.as_cudaStream_t(),
        )
    };

    wrap_error( (), status)
}


//  # Stream operations
//  # -----------------

extern "C" {
    pub fn cublasSetStream_v2(handle: cublasHandle_t, streamId: cudaStream_t) -> cublasStatus_t;
    pub fn cublasGetStream_v2(
        handle: cublasHandle_t,
        streamId: *mut cudaStream_t,
    ) -> cublasStatus_t;
}

impl cuda::Stream {
    pub fn set_active(&self, handle: &Context) -> Result<(), CublasError> {
      let status = unsafe {
        cublasSetStream_v2(handle.as_cublasHandle_t(), self.as_cudaStream_t())
      };
      wrap_error( (), status)
    }

    pub fn release(&self, handle: &Context) -> Result<(), CublasError> {
      let null = std::ptr::null_mut();
      let status = unsafe {
        cublasSetStream_v2(handle.as_cublasHandle_t(), null)
      };
      wrap_error( (), status)
    }
}

//  # BLAS
//  # ----


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

// ------------------------------------------------------------------------

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn memory() {
        let _ctx = Context::new().unwrap();
        let matrix = vec![1.0, 2.0, 3.0, 4.0,
                          1.1, 2.1, 3.1, 4.1f64];
        let mut a = vec![ 2.0f64 ; 8 ];
        let mut d_a = DevPtr::new(cuda::Device::GPU(0), matrix.len()).unwrap();
        set_matrix(4, 2, &matrix, 4, &mut d_a, 4).unwrap();
        get_matrix(4, 2, &d_a, 4, &mut a, 4).unwrap();
        assert_eq!(a, matrix);
    }
}

