use std::fmt;
use ::std::os::raw::c_uint;

pub struct CudaError(pub(crate) c_uint);

fn fmt_error(s: &CudaError, f: &mut fmt::Formatter<'_>) -> fmt::Result {
       write!(f, "Cuda error {}", s)
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


