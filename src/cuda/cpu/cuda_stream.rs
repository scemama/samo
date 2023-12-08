use super::*;

use ::std::os::raw::{c_void, c_uint};

use c_uint as cudaError_t;

#[allow(non_camel_case_types)]
pub type cudaStream_t = *mut c_void;

#[derive(Debug,Clone)]
pub struct Stream {
    handle: Box<Vec<i32>>
}

impl Stream {

    pub fn new() -> Self {
        let mut handle = Box::new(vec![1i32 ; 1]);
        Self { handle }
    }

    pub fn as_raw_mut_ptr(&self) -> *mut c_void {
        self.handle.as_ptr() as *mut c_void
    }

    pub fn as_raw_ptr(&self) -> *const c_void {
        self.handle.as_ptr() as *const c_void
    }

    pub fn as_cudaStream_t(&self) -> cudaStream_t {
        self.handle.as_ptr() as cudaStream_t
    }

    pub fn synchronize(&self) {
    }

}


