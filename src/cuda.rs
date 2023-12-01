#![allow(non_camel_case_types)]

use std::{fmt, error};
use std::ffi::CStr;
use ::std::os::raw::{c_void, c_int};
use std::marker::PhantomData;

///! This module is a minimal interface to CUDA functions

//  # Error handling
//  # --------------

pub struct CudaError(::std::os::raw::c_uint);

use ::std::os::raw::c_uint as cudaError_t;

extern "C" {
    fn cudaGetErrorString(error: cudaError_t) -> *const ::std::os::raw::c_char;
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

fn wrap_error<T>(output: T, e: cudaError_t) -> Result<T, CudaError> {
    match e {
        0 => Ok(output),
        _ => Err(CudaError(e)),
    }
}




//  # Memory management
//  # -----------------

#[link(name = "cudart")]
extern "C" {
    // Memory management
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t;
    fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> cudaError_t;
    fn cudaFree(devPtr: *mut c_void) -> cudaError_t;
    fn cudaMemset(
        devPtr: *mut c_void,
        value: c_int,
        count: usize,
    ) -> cudaError_t;
}

pub struct MemInfo {
    pub free: usize,
    pub total: usize
}

/// Returns the amount of free and total memory on the device
pub fn get_mem_info() -> Result<MemInfo, CudaError> {
  let mut free = 0;
  let mut total = 0;
  let rc = unsafe { cudaMemGetInfo(&mut free, &mut total) };
  wrap_error( MemInfo {free, total}, rc)
}

/// Pointer to memory on the device
pub struct DevPtr<T> {
    raw_ptr: *mut c_void,
    size: usize,
    _phantom: PhantomData<T>,
}

impl<T> DevPtr<T>
{

    /// Allocates memory on the device and returns a pointer
    pub fn malloc(size: usize) -> Result<Self, CudaError> {
        let mut raw_ptr = std::ptr::null_mut();
        let rc = unsafe { cudaMalloc(&mut raw_ptr, size * std::mem::size_of::<T>() ) };
        let dev_ptr = Self { raw_ptr, size, _phantom: PhantomData };
        wrap_error(dev_ptr, rc)
    }


    /// Dellocates memory on the device
    fn free(&self) -> Result<(), CudaError> {
        wrap_error( (), unsafe { cudaFree(self.raw_ptr) } )
    }

    /// Copies `count` copies of `value` on the device
    pub fn memset(&mut self, value: u8) -> Result<(), CudaError> {
        wrap_error( (), unsafe { cudaMemset(self.raw_ptr, value as c_int, self.size * std::mem::size_of::<T>()) } )
    }

    pub fn as_raw_mut_ptr(&self) -> *mut c_void {
        self.raw_ptr as *mut c_void
    }

    pub fn as_raw_ptr(&self) -> *const c_void {
        self.raw_ptr as *const c_void
    }

    pub fn offset(&self, count: isize) -> Self {
        let offset: isize = count * (std::mem::size_of::<T>() as isize);
        let new_size: usize = ( (self.size as isize) - count).try_into().unwrap();
        Self { raw_ptr: unsafe { self.raw_ptr.offset(offset) },
               size: new_size,
               _phantom: PhantomData,
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn is_null(&self) -> bool {
        self.raw_ptr == std::ptr::null_mut() && self.size == 0
    }
}

impl<T> Drop for DevPtr<T> {
    fn drop(&mut self) {
        self.free().unwrap();
    }
}

impl<T> fmt::Display for DevPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{ptr: {}, size: {}}}", self.raw_ptr as u64, self.size)
    }
}

impl<T> fmt::Debug for DevPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{ptr: {}, size: {}}}", self.raw_ptr as u64, self.size)
    }
}


//  # Device choice
//  # -------------

extern "C" {
    fn cudaSetDevice(device: c_int) -> cudaError_t;
    fn cudaGetDevice(device: *mut c_int) -> cudaError_t;
    fn cudaGetDeviceCount(count: *mut c_int) -> cudaError_t;
}

/// Select device de be used for next CUDA calls
pub fn set_device(id: usize) -> Result<(), CudaError> {
    wrap_error( (), unsafe { cudaSetDevice(id.try_into().unwrap()) } )
}

/// Return the current device used for CUDA calls
pub fn get_device() -> Result<usize, CudaError> {
    let mut id: i32 = 0;
    let rc = unsafe { cudaGetDevice(&mut id) };
    wrap_error( rc.try_into().unwrap(), rc )
}


//  # CUDA Streams
//  # ------------

pub type cudaStream_t = *mut c_void;

extern "C" {
    fn cudaStreamCreate(pStream: *mut cudaStream_t) -> cudaError_t;
    fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;
    fn cudaDeviceSynchronize() -> cudaError_t;
}

use std::rc::Rc;
use std::ptr::NonNull;

#[derive(Debug)]
pub struct CudaStream {
    handle: NonNull<c_void>,  // Adjusted type here
}

impl CudaStream {

    fn new() -> Result<Self, CudaError> {
        let mut handle = std::ptr::null_mut();
        let rc = unsafe { cudaStreamCreate(&mut handle as *mut *mut c_void) };
        NonNull::new(handle).map(|handle| Self { handle })
            .ok_or(CudaError(rc))
    }

    pub fn as_raw_mut_ptr(&self) -> *mut c_void {
        self.handle.as_ptr()
    }

    pub fn as_raw_ptr(&self) -> *const c_void {
        self.handle.as_ptr()
    }

}

impl Drop for CudaStream {
    fn drop(&mut self) {
        unsafe { cudaStreamDestroy(self.as_raw_mut_ptr() as *mut c_void) };
    }
}

#[derive(Debug, Clone)]
pub struct Stream(Rc<CudaStream>);

impl Stream {

    pub fn create() -> Result<Self, CudaError> {
        CudaStream::new().map(|context| Self(Rc::new(context)))
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
}


//-----
pub fn device_synchronize() -> Result<(), CudaError> {
     wrap_error( (), unsafe { cudaDeviceSynchronize() } )
}






// ------------------------------------------------------------------------

#[cfg(all(test,feature="cublas"))]
mod tests {

    use super::*;

    #[test]
    fn memory() {
        let info = get_mem_info().unwrap();
        println!("Free: {}\nTotal: {}", info.free, info.total);
        assert!(info.free > 0);
        assert!(info.total > 0);

        let mut dev_ptr = DevPtr::<f64>::malloc(10).unwrap();
        dev_ptr.memset(1).unwrap();
    }
}

