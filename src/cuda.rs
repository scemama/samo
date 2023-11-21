///! This module is a minimal interface to CUDA functions

//  # Error handling
//  # --------------

use std::{fmt, error};
use std::ffi::CStr;

#[derive(Debug)]
pub struct CudaError(::std::os::raw::c_uint);

use ::std::os::raw::c_uint as cudaError_t;

extern "C" {
    fn cudaGetErrorString(error: cudaError_t) -> *const ::std::os::raw::c_char;
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
       let msg : &CStr = unsafe { CStr::from_ptr(cudaGetErrorString(self.0)) };
       let msg = msg.to_str().unwrap();
       write!(f, "{}", msg)
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

/// Pointer to memory on the device
pub struct DevPtr(*mut ::std::os::raw::c_void);

extern "C" {
    // Memory management
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t;
    fn cudaMalloc(devPtr: *mut *mut ::std::os::raw::c_void, size: usize) -> cudaError_t;
    fn cudaFree(devPtr: *mut ::std::os::raw::c_void) -> cudaError_t;
    fn cudaMemset(
        devPtr: *mut ::std::os::raw::c_void,
        value: ::std::os::raw::c_int,
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


impl DevPtr {

    /// Allocates memory on the device and returns a pointer
    pub fn malloc(size: usize) -> Result<DevPtr, CudaError> {
        let mut dev_ptr = Self(std::ptr::null_mut());
        let rc = unsafe { cudaMalloc(&mut dev_ptr.0, size) };
        wrap_error(dev_ptr, rc)
    }


    /// Dellocates memory on the device
    pub fn free(&self) -> Result<(), CudaError> {
        wrap_error( (), unsafe { cudaFree(self.0) } )
    }

    /// Copies `count` copies of `value` on the device
    pub fn memset(&self, value: u32, count: usize) -> Result<(), CudaError> {
        wrap_error( (), unsafe { cudaMemset(self.0, value as ::std::os::raw::c_int, count) } )
    }

    pub fn as_raw_mut_ptr(&self) -> *mut ::std::os::raw::c_void {
        self.0
    }

    pub fn as_raw_ptr(&self) -> *const ::std::os::raw::c_void {
        self.0 as *const ::std::os::raw::c_void
    }
}



//  # Device choice
//  # -------------

extern "C" {
    fn cudaSetDevice(device: ::std::os::raw::c_int) -> cudaError_t;
    fn cudaGetDevice(device: *mut ::std::os::raw::c_int) -> cudaError_t;
    fn cudaGetDeviceCount(count: *mut ::std::os::raw::c_int) -> cudaError_t;
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

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUstream_st {
    _unused: [u8; 0],
}
pub type Stream = *mut CUstream_st;
use self::Stream as cudaStream_t;


extern "C" {
    fn cudaStreamCreate(pStream: *mut cudaStream_t) -> cudaError_t;
    fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;
    fn cudaDeviceSynchronize() -> cudaError_t;
}


