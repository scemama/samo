use std::error::Error;

// C Cbindings to the CUDA library.

type CudaError = ::std::os::raw::c_uint;
const cudaError_cudaSuccess: CudaError = 0;
use self::CudaError as cudaError_t;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUstream_st {
    _unused: [u8; 0],
}
pub type Stream = *mut CUstream_st;
use self::Stream as cudaStream_t;

pub type Ptr = *mut ::std::os::raw::c_void;

extern "C" {
    // Error handling
    fn cudaGetErrorString(error: cudaError_t) -> *const ::std::os::raw::c_char;

    // Memory management
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t;
    fn cudaMalloc(devPtr: *mut *mut ::std::os::raw::c_void, size: usize) -> cudaError_t;
    fn cudaFree(devPtr: *mut ::std::os::raw::c_void) -> cudaError_t;
    fn cudaMemset(
        devPtr: *mut ::std::os::raw::c_void,
        value: ::std::os::raw::c_int,
        count: usize,
    ) -> cudaError_t;


    // Choose device
    fn cudaSetDevice(device: ::std::os::raw::c_int) -> cudaError_t;
    fn cudaGetDevice(device: *mut ::std::os::raw::c_int) -> cudaError_t;
    fn cudaGetDeviceCount(count: *mut ::std::os::raw::c_int) -> cudaError_t;

    // CUDA streams
    fn cudaStreamCreate(pStream: *mut cudaStream_t) -> cudaError_t;
    fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;
    fn cudaDeviceSynchronize() -> cudaError_t;

}

pub fn make_error(e: cudaError_t) -> Error {
    let msg = String::from(cudaGetErrorString(e));
    Error::new("CUDA Error: {msg}")
}

struct MemInfo {
    free: usize,
    total: usize
}

pub fn cudaGetMemInfo() -> Result<MemInfo, Error> {
  let mut free = 0;
  let mut total = 0;
  match cudaMemGetInfo(&mut free, &mut total) {
    cudaError_cudaSuccess => Ok(MemInfo {free, total}),
    e => make_error(e),
  }
}



pub fn malloc(size: usize) -> Result<Ptr, Error> {
    let mut ptr: Ptr;
    match cudaMalloc(&mut ptr, size) {
        cudaError_cudaSuccess => Ok(ptr),
        e => make_error(e),
    }
}


pub fn free(ptr: &Ptr) -> Result<(), Error> {
    match cudaFree(&ptr) {
        cudaError_cudaSuccess => Ok(()),
        e => make_error(e),
    }
}

