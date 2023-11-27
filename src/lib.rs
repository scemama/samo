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
//! let matrix = TiledMatrix::<f64>::new(64, 64, 1.0);
//!
//! // Perform operations on the matrix...
//! ```

include!("common.rs");

use std::os::raw::c_char;

macro_rules! make_samo_tile {
    ($s:ty,
     $samo_tile:ident,
     $samo_untile:ident,
     $samo_free:ident,
     $samo_gemm_tiled:ident
     ) => {

        #[no_mangle]
        pub unsafe extern "C" fn $samo_tile(a: *mut $s, nrows: i64, ncols: i64, lda: i64) -> *mut TiledMatrix<$s> {
            let nrows: usize = nrows as usize;
            let ncols: usize = ncols as usize;
            let lda: usize = lda as usize;

            let a_vec = core::slice::from_raw_parts(a, lda*ncols);
            let result = TiledMatrix::<$s>::from(&a_vec, nrows, ncols, lda );
            Box::into_raw(Box::new(result))
        }

        #[no_mangle]
        pub unsafe extern "C" fn $samo_untile(a_tiled: *const TiledMatrix<$s>, a: *mut $s, lda: i64) {
            let lda: usize = lda.try_into().unwrap();
            let ncols = (*a_tiled).ncols();
            let mut a_vec = core::slice::from_raw_parts_mut(a, lda*ncols);
            (*a_tiled).copy_in_vec(&mut a_vec, lda);
        }

        #[no_mangle]
        pub unsafe extern "C" fn $samo_free(a_tiled: *mut TiledMatrix<$s>) {
            drop(Box::from_raw(a_tiled));
        }

        #[no_mangle]
        pub unsafe extern "C" fn $samo_gemm_tiled(transa: c_char, transb: c_char, alpha: $s,
            a: *const TiledMatrix<$s>, b: *const TiledMatrix<$s>, beta: $s, c: *mut TiledMatrix<$s> ) {

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

            TiledMatrix::<$s>::gemm_mut(alpha, &*a, &*b, beta, &mut *c);
        }
    }
}

make_samo_tile!(f64, samo_dtile, samo_duntile, samo_dfree, samo_dgemm_tiled);
make_samo_tile!(f32, samo_stile, samo_suntile, samo_sfree, samo_sgemm_tiled);


