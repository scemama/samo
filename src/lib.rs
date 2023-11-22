//! # Syncronous/Asynchonous Matrix Operations
//!
//! `samo` is a collection of utilities to work with matrices that are
//! too large to fit into memory efficiently and need to be divided
//! into smaller blocks or "tiles".  This approach can significantly
//! improve performance for operations on matrices by optimizing for
//! cache locality and parallel computation.
//!
//! ## Modules
//!
//! - `tile`: Contains the `Tile` struct and its associated
//!   implementations. A `Tile` is a submatrix block that represents a
//!   portion of a larger matrix.
//! - `tiled_matrix`: Contains the `TiledMatrix` struct that
//!   represents a matrix divided into tiles. This module provides the
//!   necessary functionality to work with tiled matrices as a whole.
//!
//! ## Usage
//!
//! This crate is meant to be used with numeric computations where
//! matrix operations are a performance bottleneck. By dividing a
//! matrix into tiles, operations can be performed on smaller subsets
//! of the data, which can be more cache-friendly and parallelizable.
//!
//! ### Example
//!
//! ```
//! use samo::TiledMatrix;
//!
//! // Create a 64x64 matrix and fill it with ones.
//! let matrix = TiledMatrix::new(64, 64, 1.0);
//!
//! // Perform operations on the matrix...
//! ```

include!("common.rs");

use std::os::raw::c_char;

unsafe fn samo_tile<T>(a: *mut T, nrows: i64, ncols: i64, lda: i64) -> *mut TiledMatrix<T>
where T: Float
{
    let nrows: usize = nrows as usize;
    let ncols: usize = ncols as usize;
    let lda: usize = lda as usize;

    let a_vec = core::slice::from_raw_parts(a, lda*ncols);
    let result = TiledMatrix::from(&a_vec, nrows, ncols, lda );
    Box::into_raw(Box::new(result))
}

#[no_mangle]
pub unsafe extern "C" fn samo_dtile(a: *mut f64, nrows: i64, ncols: i64, lda: i64) -> *mut TiledMatrix<f64> {
    samo_tile(a,nrows,ncols,lda)
}

#[no_mangle]
pub unsafe extern "C" fn samo_stile(a: *mut f32, nrows: i64, ncols: i64, lda: i64) -> *mut TiledMatrix<f32> {
    samo_tile(a,nrows,ncols,lda)
}



unsafe fn samo_untile<T>(a_tiled: *const TiledMatrix<T>, a: *mut T, lda: i64)
where T: Float
{
    let lda: usize = lda.try_into().unwrap();
    let ncols = (*a_tiled).ncols();
    let mut a_vec = core::slice::from_raw_parts_mut(a, lda*ncols);
    (*a_tiled).copy_in_vec(&mut a_vec, lda);
}

#[no_mangle]
pub unsafe extern "C" fn samo_duntile(a_tiled: *const TiledMatrix<f64>, a: *mut f64, lda: i64) {
    samo_untile(a_tiled, a, lda);
}

#[no_mangle]
pub unsafe extern "C" fn samo_suntile(a_tiled: *const TiledMatrix<f32>, a: *mut f32, lda: i64) {
    samo_untile(a_tiled, a, lda);
}



unsafe fn samo_free<T>(a_tiled: *mut TiledMatrix<T>)
where T: Float
{
    drop(Box::from_raw(a_tiled));
}

#[no_mangle]
pub unsafe extern "C" fn samo_dfree(a_tiled: *mut TiledMatrix<f64>) {
    samo_free(a_tiled)
}

#[no_mangle]
pub unsafe extern "C" fn samo_sfree(a_tiled: *mut TiledMatrix<f32>) {
    samo_free(a_tiled)
}



unsafe fn samo_gemm<T>(transa: c_char, transb: c_char, alpha: T,
      a: *const TiledMatrix<T>, b: *const TiledMatrix<T>, beta: T, c: *mut TiledMatrix<T> )
where T: Float
{

    let a =
        match transa as u8 {
            b'N' | b'n' => a,
            b'T' | b't' => &(*a).t(),
            _ => {panic!("transa should be ['N'|'T'], not {transa}")},
        };

    let b =
        match transb as u8 {
            b'N' | b'n' => b,
            b'T' | b't' => &(*b).t(),
            _ => {panic!("transb should be ['N'|'T'], not {transb}")},
        };

    tiled_matrix::gemm_mut(alpha, &*a, &*b, beta, &mut *c);
}

#[no_mangle]
pub unsafe extern "C" fn samo_dgemm_tiled(transa: c_char, transb: c_char, alpha: f64,
      a: *const TiledMatrix<f64>, b: *const TiledMatrix<f64>, beta: f64, c: *mut TiledMatrix<f64> ) {
    samo_gemm(transa,transb,alpha,a,b,beta,c);
}

#[no_mangle]
pub unsafe extern "C" fn samo_sgemm_tiled(transa: c_char, transb: c_char, alpha: f32,
      a: *const TiledMatrix<f32>, b: *const TiledMatrix<f32>, beta: f32, c: *mut TiledMatrix<f32> ) {
    samo_gemm(transa,transb,alpha,a,b,beta,c);
}


