use super::*;
use crate::cuda;
use crate::cuda::{Stream, cudaStream_t};
use cuda::DevPtr;
use std::os::raw::c_void;

#[link(name = "cublas")]
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


