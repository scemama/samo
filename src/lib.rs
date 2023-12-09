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
     $gemm:ident
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
        
        #[no_mangle]
        pub unsafe extern "C" fn $get_pointer(a: *mut Matrix::<$s>)
                                              -> *mut $s {
            (*a).as_slice_mut().as_mut_ptr()
        }


        /// Free a matrix allocated using the $malloc function
        #[no_mangle]
        pub unsafe extern "C" fn $free(a: *mut Matrix<$s>) {
            drop(Box::from_raw(a));
        }


        /// Matrix multiplication
        #[no_mangle]
        pub unsafe extern "C" fn $gemm (transa: c_char,
                                        transb: c_char,
                                        alpha: $s,
                                        a: *const Matrix<$s>,
                                        b: *const Matrix<$s>,
                                        beta: $s,
                                        c: *mut Matrix<$s> ) {
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
                    Matrix::<$s>::gemm_mut(alpha, &*a, &*b, beta, &mut *c);
                }, 
                (true,false) => {
                    let a = &(*a).t();
                    Matrix::<$s>::gemm_mut(alpha, &*a, &*b, beta, &mut *c);
                }, 
                (false,true) => {
                    let b = &(*b).t();
                    Matrix::<$s>::gemm_mut(alpha, &*a, &*b, beta, &mut *c);
                }, 
                (true,true) => {
                    let a = &(*a).t();
                    let b = &(*b).t();
                    Matrix::<$s>::gemm_mut(alpha, &*a, &*b, beta, &mut *c);
                }, 
            }

        }
    }
}

make_samo_matrix!(f64, samo_dmalloc, samo_dfree, samo_dget_pointer, samo_dgemm);


