use std::{fmt, error};
use std::ffi::CStr;
use ::std::os::raw::c_uint;

//  # Error handling
//  # --------------

pub struct CudaError(pub(crate) c_uint);

#[link(name = "cudart")]
extern "C" {
    fn cudaGetErrorString(error: c_uint) -> *const ::std::os::raw::c_char;
}

fn fmt_error(s: &CudaError, f: &mut fmt::Formatter<'_>) -> fmt::Result {
       let msg : &CStr = unsafe { CStr::from_ptr(cudaGetErrorString(s.0)) };
       let msg = msg.to_str().unwrap();
       write!(f, "{}", msg)
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
       fmt_error(self, f)
    }
}

impl fmt::Debug for CudaError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
       fmt_error(self, f)
  }
}

impl error::Error for CudaError {}

pub(crate) fn wrap_error<T>(output: T, e: c_uint) -> Result<T, CudaError> {
    match e {
        0 => Ok(output),
        _ => Err(CudaError(e)),
    }
}

