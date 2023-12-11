use super::*;

use std::sync::Arc;
use std::ptr::NonNull;

use ::std::os::raw::{c_void, c_uint};

use c_uint as cudaError_t;

#[allow(non_camel_case_types)]
pub type cudaStream_t = *mut c_void;

#[link(name = "cudart")]
extern "C" {
    fn cudaStreamCreate(pStream: *mut cudaStream_t) -> cudaError_t;
    fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;
    fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;
}


/// Pointer to memory on the device
#[derive(Debug)]
struct CudaStreamPtr(NonNull<c_void>);

impl CudaStreamPtr {
  fn as_ptr(&self) -> *mut c_void {
    self.0.as_ptr()
  }
}

impl Drop for CudaStreamPtr {
    fn drop(&mut self) {
        unsafe { cudaStreamDestroy(self.0.as_ptr()) };
    }
}



#[derive(Debug,Clone)]
pub struct Stream {
    handle: Arc<CudaStreamPtr>,
}

impl Stream {

    pub fn new() -> Self {
        let mut handle = std::ptr::null_mut();
        let rc = unsafe { cudaStreamCreate(&mut handle as *mut *mut c_void) };
        NonNull::new(handle).map(|handle| Self { handle: Arc::new(CudaStreamPtr(handle)) })
            .ok_or(CudaError(rc)).unwrap()
    }

    pub fn as_raw_mut_ptr(&self) -> *mut c_void {
        self.handle.as_ptr()
    }

    pub fn as_raw_ptr(&self) -> *const c_void {
        self.handle.as_ptr()
    }

    pub fn as_cudaStream_t(&self) -> cudaStream_t {
        self.handle.as_ptr()
    }

    pub fn synchronize(&self) {
        let rc = unsafe { cudaStreamSynchronize(self.handle.as_ptr()) };
        wrap_error((), rc).unwrap()
    }

}

