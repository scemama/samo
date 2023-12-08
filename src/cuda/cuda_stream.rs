#![allow(non_camel_case_types)]

use super::*;

use std::sync::Arc;
use std::ptr::NonNull;

use std::{fmt, error};
use std::ffi::CStr;
use ::std::os::raw::{c_void, c_int, c_uint};
use std::marker::PhantomData;

use c_uint as cudaError_t;

pub type cudaStream_t = *mut c_void;

#[link(name = "cudart")]
extern "C" {
    fn cudaStreamCreate(pStream: *mut cudaStream_t) -> cudaError_t;
    fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;
    fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;
}


#[derive(Debug)]
pub struct CudaStream {
    handle: NonNull<c_void>,
}

impl CudaStream {

    fn new() -> Result<Self, CudaError> {
        let mut handle = std::ptr::null_mut();
        let rc = unsafe { cudaStreamCreate(&mut handle as *mut *mut c_void) };
        NonNull::new(handle).map(|handle| Self { handle })
            .ok_or(CudaError(rc))
    }

    fn as_raw_mut_ptr(&self) -> *mut c_void {
        self.handle.as_ptr()
    }

    fn as_raw_ptr(&self) -> *const c_void {
        self.handle.as_ptr()
    }

    fn synchronize(&self) -> Result<(), CudaError> {
        let rc = unsafe { cudaStreamSynchronize(self.handle.as_ptr()) };
        wrap_error((), rc)
    }

}

impl Drop for CudaStream {
    fn drop(&mut self) {
        unsafe { cudaStreamDestroy(self.as_raw_mut_ptr() as *mut c_void) };
    }
}

#[derive(Debug, Clone)]
pub struct Stream(Arc<CudaStream>);

impl Stream {

    pub fn new() -> Self {
        CudaStream::new().map(|context| Self(Arc::new(context))).unwrap()
    }

    pub fn as_raw_mut_ptr(&self) -> *mut c_void {
        self.0.as_raw_mut_ptr()
    }

    pub fn as_raw_ptr(&self) -> *const c_void {
        self.0.as_raw_ptr()
    }

    pub fn as_cudaStream_t(&self) -> cudaStream_t {
        self.0.as_raw_mut_ptr()
    }

    pub fn synchronize(&self) {
        self.0.synchronize().unwrap()
    }
}

