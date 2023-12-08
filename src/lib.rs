//! # Syncronous/Asynchonous Matrix Operations

include!("common.rs");

use std::os::raw::c_char;
use std::thread;
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
     $free:ident
    ) => {

      /// Allocate a new matrix.
      /// device = -1: allocate on CPU
      /// device = id: allocate on GPU(id)
      #[no_mangle]
      pub unsafe extern "C" fn $malloc (device: i32, nrows: i64, ncols: i64)
      -> *mut Matrix::<$s> {
        let device = cuda::Device::new(device);
        let result = Matrix::<$s>::new(device,
                            nrows.try_into().unwrap(),
                            ncols.try_into().unwrap());
        Box::into_raw(Box::new(result))
      }

    }
}

make_samo_matrix!(f64, samo_dmalloc, samo_dfree);


