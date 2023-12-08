use super::*;

use std::sync::Arc;
use std::ptr::NonNull;

use ::std::os::raw::{c_void, c_uint};

use c_uint as cudaError_t;

pub type cudaStream_t = *mut c_void;

#[link(name = "cudart")]
extern "C" {
    fn cudaStreamCreate(pStream: *mut cudaStream_t) -> cudaError_t;
    fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;
    fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;
}


#[derive(Debug,Clone)]
pub struct Stream {
    handle: Arc<NonNull<c_void>>,
}

impl Stream {

    pub fn new() -> Self {
        let mut handle = std::ptr::null_mut();
        let rc = unsafe { cudaStreamCreate(&mut handle as *mut *mut c_void) };
        NonNull::new(handle).map(|handle| Self { handle: Arc::new(handle) })
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

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { cudaStreamDestroy(self.as_raw_mut_ptr() as *mut c_void) };
    }
}

