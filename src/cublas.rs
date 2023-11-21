///! This module is a minimal interface to CuBLAS functions

use crate::cuda;
use crate::cuda::Stream as cudaStream_t;

//  # Error handling
//  # --------------

use std::{fmt, error};

#[derive(Debug)]
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


impl fmt::Display for CublasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
       match self.0 {
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

type cublasHandle_t = *mut ::std::os::raw::c_void;
pub struct Context(cublasHandle_t);

extern "C" {
    pub fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t;
    pub fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t;
}

impl Context {
    /// Creates a new CuBLAS context
    pub fn create() -> Result<Self, CublasError> {
        let mut handle = Self(std::ptr::null_mut());
        let rc = unsafe { cublasCreate_v2(&mut handle.0) };
        wrap_error(handle, rc)
    }

    /// Destroys a CuBLAS context
    pub fn destroy(&self) -> Result<(), CublasError> {
        wrap_error( (), unsafe { cublasDestroy_v2(self.0) } )
    }
}


//  # Memory operations
//  # -----------------

extern "C" {
    pub fn cublasSetMatrix_64(
        rows: i64,
        cols: i64,
        elemSize: i64,
        A: *const ::std::os::raw::c_void,
        lda: i64,
        B: *mut ::std::os::raw::c_void,
        ldb: i64,
    ) -> cublasStatus_t;

    pub fn cublasSetMatrixAsync_64(
        rows: i64,
        cols: i64,
        elemSize: i64,
        A: *const ::std::os::raw::c_void,
        lda: i64,
        B: *mut ::std::os::raw::c_void,
        ldb: i64,
        stream: cudaStream_t,
    ) -> cublasStatus_t;

    pub fn cublasScopy_v2_64(
        handle: cublasHandle_t,
        n: i64,
        x: *const f32,
        incx: i64,
        y: *mut f32,
        incy: i64,
    ) -> cublasStatus_t;

    pub fn cublasGetMatrix_64(
        rows: i64,
        cols: i64,
        elemSize: i64,
        A: *const ::std::os::raw::c_void,
        lda: i64,
        B: *mut ::std::os::raw::c_void,
        ldb: i64,
    ) -> cublasStatus_t;
}


/// Sends an array to the device
pub fn set_matrix<T>(rows: usize, cols: usize, a: &[T], lda: usize, d_a: &mut cuda::DevPtr, d_lda: usize) -> Result<(), CublasError> {
    // Check that the slice is not empty and the dimensions are valid.
    if a.is_empty() || rows == 0 || cols == 0 {
        return Err(CublasError(INVALID_VALUE));
    }

    // Check that the slice length is at least as large as rows * lda to ensure memory safety.
    if a.len() < rows * lda {
        return Err(CublasError(INVALID_VALUE));
    }

    // Calculate the size of the elements in the slice.
    let elem_size = std::mem::size_of::<T>() as i64;

    // Perform the FFI call.
    let status = unsafe {
        cublasSetMatrix_64(
            rows as i64,
            cols as i64,
            elem_size,
            a.as_ptr() as *const ::std::os::raw::c_void,
            lda as i64,
            d_a.as_raw_mut_ptr(),
            d_lda as i64,
        )
    };

    wrap_error( (), status)
}


/// Fetches an array from the device
pub fn get_matrix<T>(rows: usize, cols: usize, d_a: &cuda::DevPtr, d_lda: usize, a: &mut [T], lda: usize) -> Result<(), CublasError> {
    // Check that the slice is not empty and the dimensions are valid.
    if a.is_empty() || rows == 0 || cols == 0 {
        return Err(CublasError(INVALID_VALUE));
    }

    // Check that the slice length is at least as large as rows * lda to ensure memory safety.
    if a.len() < rows * lda {
        return Err(CublasError(INVALID_VALUE));
    }

    // Calculate the size of the elements in the slice.
    let elem_size = std::mem::size_of::<T>() as i64;

    // Perform the FFI call.
    let status = unsafe {
        cublasGetMatrix_64(
            rows as i64,
            cols as i64,
            elem_size,
            d_a.as_raw_ptr(),
            d_lda as i64,
            a.as_mut_ptr() as *mut ::std::os::raw::c_void,
            lda as i64,
        )
    };

    wrap_error( (), status)
}

/*

//  # Stream operations
//  # -----------------

extern "C" {
    pub fn cublasSetStream_v2(handle: cublasHandle_t, streamId: cudaStream_t) -> cublasStatus_t;
    pub fn cublasGetStream_v2(
        handle: cublasHandle_t,
        streamId: *mut cudaStream_t,
    ) -> cublasStatus_t;
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
    pub fn cublasSgemv_v2_64(
        handle: cublasHandle_t,
        trans: cublasOperation_t,
        m: i64,
        n: i64,
        alpha: *const f32,
        A: *const f32,
        lda: i64,
        x: *const f32,
        incx: i64,
        beta: *const f32,
        y: *mut f32,
        incy: i64,
    ) -> cublasStatus_t;

    pub fn cublasDgemv_v2_64(
        handle: cublasHandle_t,
        trans: cublasOperation_t,
        m: i64,
        n: i64,
        alpha: *const f64,
        A: *const f64,
        lda: i64,
        x: *const f64,
        incx: i64,
        beta: *const f64,
        y: *mut f64,
        incy: i64,
    ) -> cublasStatus_t;

    pub fn cublasSgemm_v2_64(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i64,
        n: i64,
        k: i64,
        alpha: *const f32,
        A: *const f32,
        lda: i64,
        B: *const f32,
        ldb: i64,
        beta: *const f32,
        C: *mut f32,
        ldc: i64,
    ) -> cublasStatus_t;

    pub fn cublasDgemm_v2_64(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i64,
        n: i64,
        k: i64,
        alpha: *const f64,
        A: *const f64,
        lda: i64,
        B: *const f64,
        ldb: i64,
        beta: *const f64,
        C: *mut f64,
        ldc: i64,
    ) -> cublasStatus_t;

    pub fn cublasSgeam_64(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i64,
        n: i64,
        alpha: *const f32,
        A: *const f32,
        lda: i64,
        beta: *const f32,
        B: *const f32,
        ldb: i64,
        C: *mut f32,
        ldc: i64,
    ) -> cublasStatus_t;

    pub fn cublasDgeam_64(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i64,
        n: i64,
        alpha: *const f64,
        A: *const f64,
        lda: i64,
        beta: *const f64,
        B: *const f64,
        ldb: i64,
        C: *mut f64,
        ldc: i64,
    ) -> cublasStatus_t;

}

*/
