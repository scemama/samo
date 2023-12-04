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
    fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> cudaError_t;
    fn cudaFree(devPtr: *mut c_void) -> cudaError_t;
    fn cudaFreeHost(devPtr: *mut c_void) -> cudaError_t;
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
pub struct CudaDevPtr<T> {
    raw_ptr: NonNull<c_void>,
    size: usize,
    _phantom: PhantomData<T>,
}

impl<T> CudaDevPtr<T>
{

    /// Allocates memory on the device and returns a pointer
    fn new(size: usize) -> Result<Self, CudaError> {
        let mut raw_ptr = std::ptr::null_mut();
        let rc = unsafe { cudaMalloc(&mut raw_ptr as *mut *mut c_void,
                                     size * std::mem::size_of::<T>() ) };
        NonNull::new(raw_ptr).map(|raw_ptr| Self { raw_ptr, size, _phantom: PhantomData })
           .ok_or(CudaError(rc))
    }

    /// Dellocates memory on the device
    fn free(&self) -> Result<(), CudaError> {
        wrap_error( (), unsafe { cudaFree(self.raw_ptr.as_ptr()) } )
    }

    /// Copies `count` copies of `value` on the device
    fn memset(&self, value: u8) -> Result<(), CudaError> {
        wrap_error( (), unsafe {
            cudaMemset(self.raw_ptr.as_ptr(),
                       value as c_int,
                       self.size * std::mem::size_of::<T>())
            } )
    }

    fn as_raw_mut_ptr(&self) -> *mut c_void {
        self.raw_ptr.as_ptr()
    }

    fn as_raw_ptr(&self) -> *const c_void {
        self.raw_ptr.as_ptr()
    }

    fn offset(&self, count: isize) -> Self {
        let offset: isize = count * (std::mem::size_of::<T>() as isize);
        let new_size: usize = ( (self.size as isize) - count).try_into().unwrap();
        let raw_ptr = unsafe { self.raw_ptr.as_ptr().offset(offset) };
        NonNull::new(raw_ptr).map(|raw_ptr|
           Self { raw_ptr, size: new_size, _phantom: PhantomData, }).unwrap()
    }

    pub fn size(&self) -> usize {
        self.size
    }

}

impl<T> Drop for CudaDevPtr<T> {
    fn drop(&mut self) {
        self.free().unwrap();
    }
}

#[derive(Debug, Clone)]
pub struct DevPtr<T>(Arc<CudaDevPtr<T>>);

impl<T> DevPtr<T>
{

    /// Allocates memory on the device and returns a pointer
    pub fn malloc(size: usize) -> Result<Self, CudaError> {
        CudaDevPtr::new(size).map(|dev_ptr| Self(Arc::new(dev_ptr)))
    }

    /// Copies `count` copies of `value` on the device
    pub fn memset(&mut self, value: u8) -> Result<(), CudaError> {
        self.0.memset(value)
    }

    pub fn as_raw_mut_ptr(&self) -> *mut c_void {
        self.0.as_raw_mut_ptr()
    }

    pub fn as_raw_ptr(&self) -> *const c_void {
        self.0.as_raw_ptr()
    }

    pub fn offset(&self, count: isize) -> Self {
        let dev_ptr = self.0.offset(count);
        Self(Arc::new(dev_ptr))
    }

    pub fn size(&self) -> usize {
        self.0.size()
    }

}


impl<T> fmt::Display for CudaDevPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{ptr: {}, size: {}}}", self.raw_ptr.as_ptr() as u64, self.size)
    }
}

impl<T> fmt::Debug for CudaDevPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{ptr: {}, size: {}}}", self.raw_ptr.as_ptr() as u64, self.size)
    }
}


/// Pointer to pinned memory on the host
pub struct CudaHostPtr<T> {
    raw_ptr: NonNull<c_void>,
    size: usize,
    _phantom: PhantomData<T>,
}

impl<T> CudaHostPtr<T>
{

    /// Allocates pinned memory on the host and returns a pointer
    fn new(size: usize) -> Result<Self, CudaError> {
        let mut raw_ptr = std::ptr::null_mut();
        let rc = unsafe { cudaMallocHost(&mut raw_ptr as *mut *mut c_void,
                                     size * std::mem::size_of::<T>() ) };
        NonNull::new(raw_ptr).map(|raw_ptr| Self { raw_ptr, size, _phantom: PhantomData })
           .ok_or(CudaError(rc))
    }

    /// Dellocates pinned memory on the host
    fn free(&self) -> Result<(), CudaError> {
        wrap_error( (), unsafe { cudaFreeHost(self.raw_ptr.as_ptr()) } )
    }

    fn as_raw_mut_ptr(&self) -> *mut c_void {
        self.raw_ptr.as_ptr()
    }

    fn as_raw_ptr(&self) -> *const c_void {
        self.raw_ptr.as_ptr()
    }

    fn offset(&self, count: isize) -> Self {
        let offset: isize = count * (std::mem::size_of::<T>() as isize);
        let new_size: usize = ( (self.size as isize) - count).try_into().unwrap();
        let raw_ptr = unsafe { self.raw_ptr.as_ptr().offset(offset) };
        NonNull::new(raw_ptr).map(|raw_ptr|
           Self { raw_ptr, size: new_size, _phantom: PhantomData, }).unwrap()
    }

    pub fn size(&self) -> usize {
        self.size
    }

}

impl<T> Drop for CudaHostPtr<T> {
    fn drop(&mut self) {
        self.free().unwrap();
    }
}

#[derive(Debug, Clone)]
pub struct HostPtr<T>(Arc<CudaHostPtr<T>>);

impl<T> HostPtr<T>
{

    /// Allocates pinned memory on the host and returns a pointer
    pub fn malloc(size: usize) -> Result<Self, CudaError> {
        CudaHostPtr::new(size).map(|dev_ptr| Self(Arc::new(dev_ptr)))
    }

    pub fn as_slice_mut(&self) -> &mut [T] {
         unsafe { std::slice::from_raw_parts_mut(self.0.as_raw_mut_ptr() as *mut T, self.0.size) }
    }

    pub fn as_slice(&self) -> &[T] {
         unsafe { std::slice::from_raw_parts(self.0.as_raw_ptr() as *const T, self.0.size) }
    }

    pub fn as_raw_mut_ptr(&self) -> *mut c_void {
        self.0.as_raw_mut_ptr()
    }

    pub fn as_raw_ptr(&self) -> *const c_void {
        self.0.as_raw_ptr()
    }

    pub fn offset(&self, count: isize) -> Self {
        let dev_ptr = self.0.offset(count);
        Self(Arc::new(dev_ptr))
    }

    pub fn size(&self) -> usize {
        self.0.size()
    }

}


impl<T> fmt::Display for CudaHostPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{ptr: {}, size: {}}}", self.raw_ptr.as_ptr() as u64, self.size)
    }
}

impl<T> fmt::Debug for CudaHostPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{ptr: {}, size: {}}}", self.raw_ptr.as_ptr() as u64, self.size)
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
    wrap_error( id.try_into().unwrap(), rc )
}

/// Return the number of devices used for CUDA calls
pub fn get_device_count() -> Result<usize, CudaError> {
    let mut id: i32 = 0;
    let rc = unsafe { cudaGetDeviceCount(&mut id) };
    wrap_error( id.try_into().unwrap(), rc )
}


//  # CUDA Streams
//  # ------------

pub type cudaStream_t = *mut c_void;

extern "C" {
    fn cudaStreamCreate(pStream: *mut cudaStream_t) -> cudaError_t;
    fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;
    fn cudaDeviceSynchronize() -> cudaError_t;
    fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;
}

use std::sync::Arc;
use std::ptr::NonNull;

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

    pub fn create() -> Result<Self, CudaError> {
        CudaStream::new().map(|context| Self(Arc::new(context)))
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

    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.0.synchronize()
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

