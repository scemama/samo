//! # Syncronous/Asynchonous Matrix Operations

include!("common.rs");

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


            /// Free a matrix allocated using the $malloc function or $submatrix
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
                                              ncols: i64) -> *mut Matrix<$s> {
                    let result = (*a).reshape(nrows.try_into().unwrap(), ncols.try_into().unwrap());
                    Box::into_raw(Box::new(result))
            }



            /// Matrix multiplication
            #[no_mangle]
            pub unsafe extern "C" fn $gemm_nn ( alpha: $s,
                                                a: *const Matrix<$s>,
                                                b: *const Matrix<$s>,
                                                beta: $s,
                                                c: *mut Matrix<$s> ) {
                Stream::<$s>::run(None,
                  Task::Gemm(TransAB::NN, alpha, a, b, beta, c)
                );
            }

            #[no_mangle]
            pub unsafe extern "C" fn $gemm_tn ( alpha: $s,
                                                a: *const Matrix<$s>,
                                                b: *const Matrix<$s>,
                                                beta: $s,
                                                c: *mut Matrix<$s> ) {
                Stream::<$s>::run(None,
                  Task::Gemm(TransAB::TN, alpha, a, b, beta, c)
                );
            }

            #[no_mangle]
            pub unsafe extern "C" fn $gemm_nt ( alpha: $s,
                                                a: *const Matrix<$s>,
                                                b: *const Matrix<$s>,
                                                beta: $s,
                                                c: *mut Matrix<$s> ) {
                Stream::<$s>::run(None,
                  Task::Gemm(TransAB::NT, alpha, a, b, beta, c)
                );
            }

            #[no_mangle]
            pub unsafe extern "C" fn $gemm_tt ( alpha: $s,
                                                a: *const Matrix<$s>,
                                                b: *const Matrix<$s>,
                                                beta: $s,
                                                c: *mut Matrix<$s> ) {
                Stream::<$s>::run(None,
                  Task::Gemm(TransAB::TT, alpha, a, b, beta, c)
                );
            }


            /// Matrix Add
            #[no_mangle]
            pub unsafe extern "C" fn $geam_nn ( alpha: $s,
                                                a: *const Matrix<$s>,
                                                beta: $s,
                                                b: *const Matrix<$s>,
                                                c: *mut Matrix<$s> ) {
                Stream::<$s>::run(None,
                  Task::Geam(TransAB::NN, alpha, a, beta, b, c)
                );
            }

            #[no_mangle]
            pub unsafe extern "C" fn $geam_tn ( alpha: $s,
                                                a: *const Matrix<$s>,
                                                beta: $s,
                                                b: *const Matrix<$s>,
                                                c: *mut Matrix<$s> ) {
                Stream::<$s>::run(None,
                  Task::Geam(TransAB::TN, alpha, a, beta, b, c)
                );
            }

            #[no_mangle]
            pub unsafe extern "C" fn $geam_nt ( alpha: $s,
                                                a: *const Matrix<$s>,
                                                beta: $s,
                                                b: *const Matrix<$s>,
                                                c: *mut Matrix<$s> ) {
                Stream::<$s>::run(None,
                  Task::Geam(TransAB::NT, alpha, a, beta, b, c)
                );
            }

            #[no_mangle]
            pub unsafe extern "C" fn $geam_tt ( alpha: $s,
                                                a: *const Matrix<$s>,
                                                beta: $s,
                                                b: *const Matrix<$s>,
                                                c: *mut Matrix<$s> ) {
                Stream::<$s>::run(None,
                  Task::Geam(TransAB::TT, alpha, a, beta, b, c)
                );
            }

        }
}

make_samo_matrix!(f64, samo_dmalloc, samo_dfree, samo_dget_pointer, samo_dcopy, samo_dsubmatrix, samo_dreshape,
    samo_dgemm_nn, samo_dgemm_tn, samo_dgemm_nt, samo_dgemm_tt,
    samo_dgeam_nn, samo_dgeam_tn, samo_dgeam_nt, samo_dgeam_tt);



use stream::Task;
use stream::TransAB;

macro_rules! make_samo_matrix_async {
    ($s:ty,
     $create:ident,
     $wait:ident,
     $free:ident,
     $gemm_nn:ident,
     $gemm_tn:ident,
     $gemm_nt:ident,
     $gemm_tt:ident,
     $geam_nn:ident,
     $geam_tn:ident,
     $geam_nt:ident,
     $geam_tt:ident
    ) => {

            #[no_mangle]
            pub unsafe extern "C" fn $create(device: i32) -> *mut Stream<$s> {
                let device = cuda::Device::new(device);
                let result = Stream::<$s>::new(device);
                Box::into_raw(Box::new(result))
            }

            #[no_mangle]
            pub unsafe extern "C" fn $wait(stream: *mut Stream<$s>) {
                let stream = Box::from_raw(stream);
                stream.wait();
            }

            /// Free the matrix
            #[no_mangle]
            pub unsafe extern "C" fn $free(stream: *mut Stream<$s>, a: *mut Matrix<$s>) {
                (*stream).push(
                  Task::Free(a)
                  );
            }

            /// Matrix multiplication
            #[no_mangle]
            pub unsafe extern "C" fn $gemm_nn (stream: *mut Stream<$s>,  alpha: $s,
                                                a: *const Matrix<$s>,
                                                b: *const Matrix<$s>,
                                                beta: $s,
                                                c: *mut Matrix<$s> ) {
                (*stream).push(
                  Task::Gemm(TransAB::NN, alpha, a, b, beta, c)
                  );
            }

            #[no_mangle]
            pub unsafe extern "C" fn $gemm_tn (stream: *mut Stream<$s>,  alpha: $s,
                                                a: *const Matrix<$s>,
                                                b: *const Matrix<$s>,
                                                beta: $s,
                                                c: *mut Matrix<$s> ) {
                (*stream).push(
                  Task::Gemm(TransAB::TN, alpha, a, b, beta, c)
                  );
            }

            #[no_mangle]
            pub unsafe extern "C" fn $gemm_nt (stream: *mut Stream<$s>,  alpha: $s,
                                                a: *const Matrix<$s>,
                                                b: *const Matrix<$s>,
                                                beta: $s,
                                                c: *mut Matrix<$s> ) {
                (*stream).push(
                  Task::Gemm(TransAB::NT, alpha, a, b, beta, c)
                  );
            }

            #[no_mangle]
            pub unsafe extern "C" fn $gemm_tt (stream: *mut Stream<$s>,  alpha: $s,
                                                a: *const Matrix<$s>,
                                                b: *const Matrix<$s>,
                                                beta: $s,
                                                c: *mut Matrix<$s> ) {
                (*stream).push(
                  Task::Gemm(TransAB::NT, alpha, a, b, beta, c)
                  );
            }


            /// Matrix Add
            #[no_mangle]
            pub unsafe extern "C" fn $geam_nn (stream: *mut Stream<$s>,  alpha: $s,
                                                a: *const Matrix<$s>,
                                                beta: $s,
                                                b: *const Matrix<$s>,
                                                c: *mut Matrix<$s> ) {
                (*stream).push(
                  Task::Geam(TransAB::NN, alpha, a, beta, b, c)
                  );
            }

            #[no_mangle]
            pub unsafe extern "C" fn $geam_tn (stream: *mut Stream<$s>,  alpha: $s,
                                                a: *const Matrix<$s>,
                                                beta: $s,
                                                b: *const Matrix<$s>,
                                                c: *mut Matrix<$s> ) {
                (*stream).push(
                  Task::Geam(TransAB::TN, alpha, a, beta, b, c)
                  );
            }

            #[no_mangle]
            pub unsafe extern "C" fn $geam_nt (stream: *mut Stream<$s>,  alpha: $s,
                                                a: *const Matrix<$s>,
                                                beta: $s,
                                                b: *const Matrix<$s>,
                                                c: *mut Matrix<$s> ) {
                (*stream).push(
                  Task::Geam(TransAB::NT, alpha, a, beta, b, c)
                  );
            }

            #[no_mangle]
            pub unsafe extern "C" fn $geam_tt (stream: *mut Stream<$s>,  alpha: $s,
                                                a: *const Matrix<$s>,
                                                beta: $s,
                                                b: *const Matrix<$s>,
                                                c: *mut Matrix<$s> ) {
                (*stream).push(
                  Task::Geam(TransAB::TT, alpha, a, beta, b, c)
                  );
            }

        }
}

make_samo_matrix_async!(f64, samo_dstream_create, samo_dstream_wait, samo_dfree_async,
    samo_dgemm_nn_async, samo_dgemm_tn_async, samo_dgemm_nt_async, samo_dgemm_tt_async,
    samo_dgeam_nn_async, samo_dgeam_tn_async, samo_dgeam_nt_async, samo_dgeam_tt_async);


