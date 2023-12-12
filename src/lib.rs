//! # Syncronous/Asynchonous Matrix Operations

include!("common.rs");

use std::os::raw::c_char;
use std::thread::JoinHandle;


#[no_mangle]
pub unsafe extern "C" fn samo_await(handle: *mut JoinHandle<()>) {
    let handle = Box::from_raw(handle);
    handle.join().unwrap();
}

#[no_mangle]
pub unsafe extern "C" fn samo_get_device_count() -> i32 {
    cuda::get_device_count().try_into().unwrap()
}

// Matrices

macro_rules! make_samo_matrix {
    ($s:ty,
     $malloc:ident,
     $free:ident,
     $get_pointer: ident,
     $copy: ident,
     $gemm_nn:ident,
     $gemm_tn:ident,
     $gemm_nt:ident,
     $gemm_tt:ident
    ) => {

            /// Allocate a new matrix.
            /// device = -1: allocate on CPU
            /// device = id: allocate on GPU(id)
            #[no_mangle]
            pub unsafe extern "C" fn $malloc (device: i32,
                                            nrows: i64,
                                            ncols: i64)
                                            -> *mut Matrix::<$s> {

                let device = cuda::Device::new(device);
                let result = Matrix::<$s>::new(device,
                                            nrows.try_into().unwrap(),
                                            ncols.try_into().unwrap());
                Box::into_raw(Box::new(result))

            }


            /// Free a matrix allocated using the $malloc function
            #[no_mangle]
            pub unsafe extern "C" fn $free(a: *mut Matrix<$s>) {
                drop(Box::from_raw(a));
            }


            /// Returns a pointer to the matrix elements
            #[no_mangle]
            pub unsafe extern "C" fn $get_pointer(a: *mut Matrix::<$s>)
                                                -> *mut $s {
                (*a).as_slice_mut().as_mut_ptr()
            }


            /// Copy a matrix into another one
            #[no_mangle]
            pub unsafe extern "C" fn $copy(source: *mut Matrix::<$s>, destination: *mut Matrix::<$s>, ) {
                (*destination).copy(&*source)
            }



            /// Matrix multiplication
            #[no_mangle]
            pub unsafe extern "C" fn $gemm_nn ( alpha: $s,
                                                a: *const Matrix<$s>,
                                                b: *const Matrix<$s>,
                                                beta: $s,
                                                c: *mut Matrix<$s> ) {
                Matrix::<$s>::gemm_mut(alpha, &*a, &*b, beta, &mut *c);
            }

            #[no_mangle]
            pub unsafe extern "C" fn $gemm_tn ( alpha: $s,
                                                a: *const Matrix<$s>,
                                                b: *const Matrix<$s>,
                                                beta: $s,
                                                c: *mut Matrix<$s> ) {
                let a = &(*a).t();
                Matrix::<$s>::gemm_mut(alpha, &*a, &*b, beta, &mut *c);
            }

            #[no_mangle]
            pub unsafe extern "C" fn $gemm_nt ( alpha: $s,
                                                a: *const Matrix<$s>,
                                                b: *const Matrix<$s>,
                                                beta: $s,
                                                c: *mut Matrix<$s> ) {
                let b = &(*b).t();
                Matrix::<$s>::gemm_mut(alpha, &*a, &*b, beta, &mut *c);
            }

            #[no_mangle]
            pub unsafe extern "C" fn $gemm_tt ( alpha: $s,
                                                a: *const Matrix<$s>,
                                                b: *const Matrix<$s>,
                                                beta: $s,
                                                c: *mut Matrix<$s> ) {
                let a = &(*a).t();
                let b = &(*b).t();
                Matrix::<$s>::gemm_mut(alpha, &*a, &*b, beta, &mut *c);
            }

        }
}

make_samo_matrix!(f64, samo_dmalloc, samo_dfree, samo_dget_pointer, samo_dcopy, samo_dgemm_nn, samo_dgemm_tn, samo_dgemm_nt, samo_dgemm_tt);


