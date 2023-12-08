use super::*;

use std::sync::Arc;
use std::ptr::NonNull;
use std::fmt;
use ::std::os::raw::{c_void, c_int, c_uint};
use std::marker::PhantomData;

use c_uint as cudaError_t;


#[link(name = "cudart")]
extern "C" {
    // Memory management
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t;
    fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> cudaError_t;
    fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> cudaError_t;
    fn cudaMallocManaged(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> cudaError_t;
    pub fn cudaMemPrefetchAsync(
        devPtr: *const ::std::os::raw::c_void,
        count: usize,
        dstDevice: ::std::os::raw::c_int,
        stream: cudaStream_t,
    ) -> cudaError_t;
    fn cudaFree(devPtr: *mut c_void) -> cudaError_t;
    fn cudaFreeHost(devPtr: *mut c_void) -> cudaError_t;
    fn cudaMemset(
        devPtr: *mut c_void,
        value: c_int,
        count: usize,
    ) -> cudaError_t;
    fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_uint
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
#[derive(Clone)]
pub struct DevPtr<T> {
    raw_ptr: Arc<NonNull<c_void>>,
    size: usize,
    device: Device,
    stream: Stream,
    _phantom: PhantomData<T>,
}

impl<T> DevPtr<T>
{

    /// Allocates memory on the device and returns a pointer
    pub fn new(device: Device, size: usize) -> Result<Self, CudaError> {
        device.set();
        let mut raw_ptr = std::ptr::null_mut();
        let stream = Stream::new();
        let rc = unsafe { cudaMallocManaged(&mut raw_ptr as *mut *mut c_void,
                                     size * std::mem::size_of::<T>(), 1  ) };
        NonNull::new(raw_ptr).map(|raw_ptr|
        Self { raw_ptr: Arc::new(raw_ptr), device, size, stream, _phantom: PhantomData })
           .ok_or(CudaError(rc))
    }

    pub fn bytes(&self) -> usize {
        self.size * std::mem::size_of::<T>()
    }

    pub fn prefetch(&mut self, device: Device) {
      self.device = device;
      wrap_error( (), unsafe {
        cudaMemPrefetchAsync(self.raw_ptr.as_ptr(), self.bytes(), device.id(),
              self.stream.as_cudaStream_t()) }).unwrap()
    }


    /// Dellocates memory on the device
    fn free(&mut self) {
        wrap_error( (), unsafe { cudaFree(self.raw_ptr.as_ptr()) } ).unwrap()
    }

    /// Copies `count` copies of `value` on the device
    pub fn memset(&mut self, value: u8) {
        self.device.set();
        wrap_error( (), unsafe {
            cudaMemset(self.raw_ptr.as_ptr(), value as c_int, self.bytes())
            } ).unwrap()
    }

    pub fn memcpy(&mut self, other: &Self) {
        self.device.set();
        assert!(self.size == other.size);
        wrap_error( (), unsafe {
            cudaMemcpy(self.raw_ptr.as_ptr(),
                       other.raw_ptr.as_ptr(),
                       self.bytes(), 4)
            } ).unwrap()
    }

    pub fn as_raw_mut_ptr(&self) -> *mut c_void {
        self.raw_ptr.as_ptr()
    }

    pub fn as_raw_ptr(&self) -> *const c_void {
        self.raw_ptr.as_ptr()
    }

    pub fn offset(&self, count: isize) -> Self {
        let offset: isize = count * (std::mem::size_of::<T>() as isize);
        let new_size: usize = ( (self.size as isize) - count).try_into().unwrap();
        let raw_ptr = unsafe { self.raw_ptr.as_ptr().offset(offset) };
        NonNull::new(raw_ptr).map(|raw_ptr|
           Self { raw_ptr: Arc::new(raw_ptr), device: self.device, stream: self.stream.clone(), size: new_size, _phantom: PhantomData, }).unwrap()
    }

#[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{ptr: {:?}, device: {}, size: {}}}",
        self.raw_ptr.as_ptr(),
        self.device,
        self.size)
    }
}

impl<T> Drop for DevPtr<T> {
    fn drop(&mut self) {
        self.free();
    }
}

unsafe impl<T> Sync for DevPtr<T> {}
unsafe impl<T> Send for DevPtr<T> {}

impl<T> fmt::Display for DevPtr<T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    self.fmt(f)
  }
}

impl<T> fmt::Debug   for DevPtr<T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    self.fmt(f)
  }
}


// ------------------------------------------------------------------------

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

