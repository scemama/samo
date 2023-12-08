use super::*;
use std::os::raw::c_void;
use std::sync::{Arc};
use std::ptr::NonNull;

#[allow(non_camel_case_types)]
pub type cublasHandle_t = *mut c_void;

#[link(name = "cublas")]
extern "C" {
    pub fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t;
    pub fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t;
}


#[derive(Debug)]
pub struct CublasContext {
    handle: NonNull<c_void>,
}

impl CublasContext {

    fn new() -> Result<Self, CublasError> {
        let mut handle = std::ptr::null_mut();
        let rc = unsafe { cublasCreate_v2(&mut handle as *mut *mut c_void) };
        NonNull::new(handle).map(|handle| Self { handle })
            .ok_or(CublasError(rc))
    }

    fn as_raw_mut_ptr(&self) -> *mut c_void {
        self.handle.as_ptr()
    }

    pub fn as_raw_ptr(&self) -> *const c_void {
        self.handle.as_ptr()
    }
}

impl Drop for CublasContext {
    fn drop(&mut self) {
        unsafe { cublasDestroy_v2(self.as_raw_mut_ptr() as *mut c_void) };
    }
}

#[derive(Debug, Clone)]
pub struct Context(Arc<CublasContext>);

impl Context {

    pub fn new() -> Result<Self, CublasError> {
        CublasContext::new().map(|context| Self(Arc::new(context)))
    }

    pub fn as_cublasHandle_t(&self) -> cublasHandle_t {
        self.0.as_raw_mut_ptr()
    }

    pub fn as_raw_ptr(&self) -> *const c_void {
        self.0.as_raw_ptr()
    }
}


