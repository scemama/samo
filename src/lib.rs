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
use std::thread;
use std::thread::JoinHandle;


macro_rules! make_samo_tile {
    ($s:ty,
     $tiled_matrix:ident,
     $samo_tile:ident,
     $samo_reshape:ident,
     $samo_untile:ident,
     $samo_free:ident,
     $samo_gemm_tiled:ident,
     $samo_untile_async:ident,
     $samo_gemm_tiled_async:ident
    ) => {

// Synchronous API

        #[no_mangle]
        pub unsafe extern "C" fn $samo_tile(a: *const $s, nrows: i64, ncols: i64, lda: i64) -> *mut $tiled_matrix<$s> {
            let nrows: usize = nrows as usize;
            let ncols: usize = ncols as usize;
            let lda: usize = lda as usize;

            let a_vec = core::slice::from_raw_parts(a, lda*ncols);
            let result = $tiled_matrix::<$s>::from(&a_vec, nrows, ncols, lda );
            Box::into_raw(Box::new(result))
        }

        #[no_mangle]
        pub unsafe extern "C" fn $samo_reshape(a: *const $tiled_matrix<$s>, nrows: i64, ncols: i64) -> *mut $tiled_matrix<$s> {
            let nrows: usize = nrows as usize;
            let ncols: usize = ncols as usize;
            let result = (*a).reshape(nrows, ncols);
            Box::into_raw(Box::new(result))
        }

        #[no_mangle]
        pub unsafe extern "C" fn $samo_untile(a_tiled: *const $tiled_matrix<$s>, a: *mut $s, lda: i64) {
            let lda: usize = lda.try_into().unwrap();
            let ncols = (*a_tiled).ncols();
            let mut a_vec = core::slice::from_raw_parts_mut(a, lda*ncols);
            (*a_tiled).copy_in_vec(&mut a_vec, lda);
        }

        #[no_mangle]
        pub unsafe extern "C" fn $samo_free(a_tiled: *mut $tiled_matrix<$s>) {
            drop(Box::from_raw(a_tiled));
        }

        #[no_mangle]
        pub unsafe extern "C" fn $samo_gemm_tiled(transa: c_char, transb: c_char, alpha: $s,
            a: *const $tiled_matrix<$s>, b: *const $tiled_matrix<$s>, beta: $s, c: *mut $tiled_matrix<$s> ) {

            let ta = 
                match transa as u8 {
                    b'N' | b'n' => false,
                    b'T' | b't' => true,
                    _ => {panic!("transa should be ['N'|'T'], not {transa}")},
                };

            let tb = 
                match transb as u8 {
                    b'N' | b'n' => false,
                    b'T' | b't' => true,
                    _ => {panic!("transa should be ['N'|'T'], not {transb}")},
                };

            match (ta, tb) {
               (false,false) => {
                          $tiled_matrix::<$s>::gemm_mut(alpha, &*a, &*b, beta, &mut *c);
                        }, 
               (true,false) => {
                          let a = &(*a).t();
                          $tiled_matrix::<$s>::gemm_mut(alpha, &*a, &*b, beta, &mut *c);
                        }, 
               (false,true) => {
                          let b = &(*b).t();
                          $tiled_matrix::<$s>::gemm_mut(alpha, &*a, &*b, beta, &mut *c);
                        }, 
               (true,true) => {
                          let a = &(*a).t();
                          let b = &(*b).t();
                          $tiled_matrix::<$s>::gemm_mut(alpha, &*a, &*b, beta, &mut *c);
                        }, 
            }
}

// Asynchronous API

        #[no_mangle]
        pub unsafe extern "C" fn $samo_untile_async(a_tiled: *const $tiled_matrix<$s>, a: *mut $s, lda: i64) -> *mut JoinHandle<()> {

          let a_tiled = &*a_tiled;
          let a = &mut *a;
          let handle = thread::Builder::new().name("samo".to_string()).spawn(move || {
             $samo_untile(a_tiled, a, lda)
          }).unwrap();

          Box::into_raw(Box::new(handle))
        }

        #[no_mangle]
        pub unsafe extern "C" fn $samo_gemm_tiled_async(transa: c_char, transb: c_char,
            alpha: $s, a: *const $tiled_matrix<$s>, b: *const $tiled_matrix<$s>,
            beta: $s, c: *mut $tiled_matrix<$s> ) -> *mut JoinHandle<()>
        {
          let a = &*a;
          let b = &*b;
          let c = &mut *c;
          let handle = thread::Builder::new().name("samo".to_string()).spawn(move || {
              $samo_gemm_tiled(transa, transb, alpha, a, b, beta, c);
          }).unwrap();
          Box::into_raw(Box::new(handle))
        }

    }
}

#[no_mangle]
pub unsafe extern "C" fn samo_await(handle: *mut JoinHandle<()>) {
  let handle = Box::from_raw(handle);
  handle.join().unwrap();
}

#[no_mangle]
pub unsafe extern "C" fn samo_get_device() -> i32 {
  let dev = cuda::get_device().unwrap();
  dev.id()
}

#[no_mangle]
pub unsafe extern "C" fn samo_get_device_count() -> i32 {
  cuda::get_device_count().unwrap().try_into().unwrap()
}

#[no_mangle]
pub unsafe extern "C" fn samo_set_device(id: i32) {
  let count: i32 = cuda::get_device_count().unwrap().try_into().unwrap();
  let dev = 
      if id < count {
        cuda::Device::new(id)
      } else {
        cuda::Device::new(id % count)
      };
  dev.set_device().unwrap()
}

make_samo_tile!(f64, TiledMatrix,
  samo_dtile, samo_dreshape, samo_duntile, samo_dfree, samo_dgemm_tiled,
  samo_duntile_async, samo_dgemm_tiled_async);

make_samo_tile!(f32, TiledMatrix,
  samo_stile, samo_sreshape, samo_suntile, samo_sfree, samo_sgemm_tiled,
  samo_suntile_async, samo_sgemm_tiled_async);

make_samo_tile!(f64, TiledMatrixGPU,
  samo_dtile_gpu, samo_dreshape_gpu, samo_duntile_gpu, samo_dfree_gpu, samo_dgemm_tiled_gpu,
  samo_duntile_gpu_async, samo_dgemm_tiled_gpu_async);

make_samo_tile!(f32, TiledMatrixGPU,
  samo_stile_gpu, samo_sreshape_gpu, samo_suntile_gpu, samo_sfree_gpu, samo_sgemm_tiled_gpu,
  samo_suntile_gpu_async, samo_sgemm_tiled_gpu_async);


