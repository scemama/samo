//! # Syncronous/Asynchonous Matrix Operations

include!("common.rs");

use std::os::raw::c_char;
use std::thread::JoinHandle;
use std::sync::Mutex;

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
     $submatrix: ident,
     $reshape: ident,
     $gemm_nn:ident,
     $gemm_tn:ident,
     $gemm_nt:ident,
     $gemm_tt:ident,
     $geam_nn:ident,
     $geam_tn:ident,
     $geam_nt:ident,
     $geam_tt:ident
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

            /// Create a sub-matrix
            #[no_mangle]
            pub unsafe extern "C" fn $submatrix(source: *mut Matrix::<$s>,
                      init_rows: i64, init_cols: i64, nrows: i64, ncols: i64) -> *mut Matrix::<$s> {
                let result = (*source).submatrix(init_rows.try_into().unwrap(), init_cols.try_into().unwrap(),
                                                 nrows.try_into().unwrap(), ncols.try_into().unwrap());
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


            /// Reshape the matrix
            #[no_mangle]
            pub unsafe extern "C" fn $reshape(a: *mut Matrix<$s>,
                                              nrows: i64,
                                              ncols: i64) {
                (*a).reshape(nrows.try_into().unwrap(),
                             ncols.try_into().unwrap())
            }



            /// Matrix multiplication
            #[no_mangle]
            pub unsafe extern "C" fn $gemm_nn ( alpha: $s,
                                                a: *const Matrix<$s>,
                                                b: *const Matrix<$s>,
                                                beta: $s,
                                                c: *mut Matrix<$s> ) {
                Matrix::<$s>::gemm_mut(None, alpha, &*a, &*b, beta, &mut *c);
            }

            #[no_mangle]
            pub unsafe extern "C" fn $gemm_tn ( alpha: $s,
                                                a: *const Matrix<$s>,
                                                b: *const Matrix<$s>,
                                                beta: $s,
                                                c: *mut Matrix<$s> ) {
                let a = &(*a).t();
                Matrix::<$s>::gemm_mut(None, alpha, &*a, &*b, beta, &mut *c);
            }

            #[no_mangle]
            pub unsafe extern "C" fn $gemm_nt ( alpha: $s,
                                                a: *const Matrix<$s>,
                                                b: *const Matrix<$s>,
                                                beta: $s,
                                                c: *mut Matrix<$s> ) {
                let b = &(*b).t();
                Matrix::<$s>::gemm_mut(None, alpha, &*a, &*b, beta, &mut *c);
            }

            #[no_mangle]
            pub unsafe extern "C" fn $gemm_tt ( alpha: $s,
                                                a: *const Matrix<$s>,
                                                b: *const Matrix<$s>,
                                                beta: $s,
                                                c: *mut Matrix<$s> ) {
                let a = &(*a).t();
                let b = &(*b).t();
                Matrix::<$s>::gemm_mut(None, alpha, &*a, &*b, beta, &mut *c);
            }


            /// Matrix Add
            #[no_mangle]
            pub unsafe extern "C" fn $geam_nn ( alpha: $s,
                                                a: *const Matrix<$s>,
                                                beta: $s,
                                                b: *const Matrix<$s>,
                                                c: *mut Matrix<$s> ) {
                Matrix::<$s>::geam_mut(None, alpha, &*a, beta, &*b, &mut *c);
            }

            #[no_mangle]
            pub unsafe extern "C" fn $geam_tn ( alpha: $s,
                                                a: *const Matrix<$s>,
                                                beta: $s,
                                                b: *const Matrix<$s>,
                                                c: *mut Matrix<$s> ) {
                let a = &(*a).t();
                Matrix::<$s>::geam_mut(None, alpha, &*a, beta, &*b, &mut *c);
            }

            #[no_mangle]
            pub unsafe extern "C" fn $geam_nt ( alpha: $s,
                                                a: *const Matrix<$s>,
                                                beta: $s,
                                                b: *const Matrix<$s>,
                                                c: *mut Matrix<$s> ) {
                let b = &(*b).t();
                Matrix::<$s>::geam_mut(None, alpha, &*a, beta, &*b, &mut *c);
            }

            #[no_mangle]
            pub unsafe extern "C" fn $geam_tt ( alpha: $s,
                                                a: *const Matrix<$s>,
                                                beta: $s,
                                                b: *const Matrix<$s>,
                                                c: *mut Matrix<$s> ) {
                let a = &(*a).t();
                let b = &(*b).t();
                Matrix::<$s>::geam_mut(None, alpha, &*a, beta, &*b, &mut *c);
            }

        }
}

make_samo_matrix!(f64, samo_dmalloc, samo_dfree, samo_dget_pointer, samo_dcopy, samo_dsubmatrix, samo_dreshape,
    samo_dgemm_nn, samo_dgemm_tn, samo_dgemm_nt, samo_dgemm_tt,
    samo_dgeam_nn, samo_dgeam_tn, samo_dgeam_nt, samo_dgeam_tt);



#[no_mangle]
pub unsafe extern "C" fn samo_stream_create(device: i32) -> *mut Stream {
    let device = cuda::Device::new(device);
    let result =
        match device {
            cuda::Device::CPU => Stream::new_cpu(),
            cuda::Device::GPU(_) => Stream::new_gpu(),
        };
    Box::into_raw(Box::new(result))
}

#[no_mangle]
pub unsafe extern "C" fn samo_stream_wait(stream: *mut Stream) -> *mut Stream {
    let stream = Box::from_raw(stream);
    Box::into_raw(Box::new( stream.wait() ))
}

macro_rules! make_samo_matrix_async {
    ($s:ty,
     $reshape: ident,
     $gemm_nn:ident,
     $gemm_tn:ident,
     $gemm_nt:ident,
     $gemm_tt:ident,
     $geam_nn:ident,
     $geam_tn:ident,
     $geam_nt:ident,
     $geam_tt:ident
    ) => {

            /// Reshape the matrix
            #[no_mangle]
            pub unsafe extern "C" fn $reshape(stream: *mut Stream,
                                              a: *mut Matrix<$s>,
                                              nrows: i64,
                                              ncols: i64) -> *mut Stream {
                let a = Mutex::new(&mut *a);
                let stream = Box::from_raw(stream);
                let result = (*stream).push(move |_| {
                    let mut a = a.lock().unwrap();
                    a.reshape(nrows.try_into().unwrap(),
                                ncols.try_into().unwrap())
                }) ;
                Box::into_raw(Box::new(result))
            }



            /// Matrix multiplication
            #[no_mangle]
            pub unsafe extern "C" fn $gemm_nn (stream: *mut Stream,  alpha: $s,
                                                a: *const Matrix<$s>,
                                                b: *const Matrix<$s>,
                                                beta: $s,
                                                c: *mut Matrix<$s> ) -> *mut Stream {
//TODO
                let a = Mutex::new(&*a);
                let b = Mutex::new(&*b);
                let c = Mutex::new(&mut *c);
                let stream = Box::from_raw(stream);
                let result = (*stream).push(move |h| {
                    let a = a.lock().unwrap();
                    let b = b.lock().unwrap();
                    let mut c = c.lock().unwrap();
                    Matrix::<$s>::gemm_mut(h, alpha, &a, &b, beta, &mut c);
                });
                Box::into_raw(Box::new(result))
            }

            #[no_mangle]
            pub unsafe extern "C" fn $gemm_tn (stream: *mut Stream,  alpha: $s,
                                                a: *const Matrix<$s>,
                                                b: *const Matrix<$s>,
                                                beta: $s,
                                                c: *mut Matrix<$s> ) -> *mut Stream {
                let a = Mutex::new(&*a);
                let b = Mutex::new(&*b);
                let c = Mutex::new(&mut *c);
                let stream = Box::from_raw(stream);
                let result = (*stream).push(move |h| {
                    let a = a.lock().unwrap().t();
                    let b = b.lock().unwrap();
                    let mut c = c.lock().unwrap();
                    Matrix::<$s>::gemm_mut(h, alpha, &a, &b, beta, &mut c);
                });
                Box::into_raw(Box::new(result))
            }

            #[no_mangle]
            pub unsafe extern "C" fn $gemm_nt (stream: *mut Stream,  alpha: $s,
                                                a: *const Matrix<$s>,
                                                b: *const Matrix<$s>,
                                                beta: $s,
                                                c: *mut Matrix<$s> ) -> *mut Stream {
                let a = Mutex::new(&*a);
                let b = Mutex::new(&*b);
                let c = Mutex::new(&mut *c);
                let stream = Box::from_raw(stream);
                let result = (*stream).push(move |h| {
                    let a = a.lock().unwrap();
                    let b = b.lock().unwrap().t();
                    let mut c = c.lock().unwrap();
                    Matrix::<$s>::gemm_mut(h, alpha, &a, &b, beta, &mut c);
                });
                Box::into_raw(Box::new(result))
            }

            #[no_mangle]
            pub unsafe extern "C" fn $gemm_tt (stream: *mut Stream,  alpha: $s,
                                                a: *const Matrix<$s>,
                                                b: *const Matrix<$s>,
                                                beta: $s,
                                                c: *mut Matrix<$s> ) -> *mut Stream {
                let a = Mutex::new(&*a);
                let b = Mutex::new(&*b);
                let c = Mutex::new(&mut *c);
                let stream = Box::from_raw(stream);
                let result = (*stream).push(move |h| {
                    let a = a.lock().unwrap().t();
                    let b = b.lock().unwrap().t();
                    let mut c = c.lock().unwrap();
                    Matrix::<$s>::gemm_mut(h, alpha, &a, &b, beta, &mut c);
                });
                Box::into_raw(Box::new(result))
            }


            /// Matrix Add
            #[no_mangle]
            pub unsafe extern "C" fn $geam_nn (stream: *mut Stream,  alpha: $s,
                                                a: *const Matrix<$s>,
                                                beta: $s,
                                                b: *const Matrix<$s>,
                                                c: *mut Matrix<$s> ) -> *mut stream::Stream {
                let a = Mutex::new(&*a);
                let b = Mutex::new(&*b);
                let c = Mutex::new(&mut *c);
                let stream = Box::from_raw(stream);
                let result = (*stream).push(move |h| {
                    let a = a.lock().unwrap();
                    let b = b.lock().unwrap();
                    let mut c = c.lock().unwrap();
                    Matrix::<$s>::geam_mut(h, alpha, &a, beta, &b, &mut c);
                });
                Box::into_raw(Box::new(result))
            }

            #[no_mangle]
            pub unsafe extern "C" fn $geam_tn (stream: *mut Stream,  alpha: $s,
                                                a: *const Matrix<$s>,
                                                beta: $s,
                                                b: *const Matrix<$s>,
                                                c: *mut Matrix<$s> ) -> *mut stream::Stream {
                let a = Mutex::new(&*a);
                let b = Mutex::new(&*b);
                let c = Mutex::new(&mut *c);
                let stream = Box::from_raw(stream);
                let result = (*stream).push(move |h| {
                    let a = a.lock().unwrap().t();
                    let b = b.lock().unwrap();
                    let mut c = c.lock().unwrap();
                    Matrix::<$s>::geam_mut(h, alpha, &a, beta, &b, &mut c);
                });
                Box::into_raw(Box::new(result))
            }

            #[no_mangle]
            pub unsafe extern "C" fn $geam_nt (stream: *mut Stream,  alpha: $s,
                                                a: *const Matrix<$s>,
                                                beta: $s,
                                                b: *const Matrix<$s>,
                                                c: *mut Matrix<$s> ) -> *mut stream::Stream {
                let a = Mutex::new(&*a);
                let b = Mutex::new(&*b);
                let c = Mutex::new(&mut *c);
                let stream = Box::from_raw(stream);
                let result = (*stream).push(move |h| {
                    let a = a.lock().unwrap();
                    let b = b.lock().unwrap().t();
                    let mut c = c.lock().unwrap();
                    Matrix::<$s>::geam_mut(h, alpha, &a, beta, &b, &mut c);
                });
                Box::into_raw(Box::new(result))
            }

            #[no_mangle]
            pub unsafe extern "C" fn $geam_tt (stream: *mut Stream,  alpha: $s,
                                                a: *const Matrix<$s>,
                                                beta: $s,
                                                b: *const Matrix<$s>,
                                                c: *mut Matrix<$s> ) -> *mut stream::Stream {
                let a = Mutex::new(&*a);
                let b = Mutex::new(&*b);
                let c = Mutex::new(&mut *c);
                let stream = Box::from_raw(stream);
                let result = (*stream).push(move |h| {
                    let a = a.lock().unwrap().t();
                    let b = b.lock().unwrap().t();
                    let mut c = c.lock().unwrap();
                    Matrix::<$s>::geam_mut(h, alpha, &a, beta, &b, &mut c);
                });
                Box::into_raw(Box::new(result))

            }
        }
}

make_samo_matrix_async!(f64, samo_dreshape_async,
    samo_dgemm_nn_async, samo_dgemm_tn_async, samo_dgemm_nt_async, samo_dgemm_tt_async,
    samo_dgeam_nn_async, samo_dgeam_tn_async, samo_dgeam_nt_async, samo_dgeam_tt_async);


