use super::*;

use std::fmt;
use ::std::os::raw::{c_void, c_int, c_uint};

use c_uint as cudaError_t;

pub struct MemInfo {
    pub free: usize,
    pub total: usize
}

/// Returns the amount of free and total memory on the device
pub fn get_mem_info() -> Result<MemInfo, CudaError> {
  let mut free = 0;
  let mut total = 0;
  Ok( MemInfo {free, total} )
}


/// Pointer to memory on the device
#[derive(Clone)]
pub struct DevPtr<T> {
    raw_ptr: Box::<Vec<T>>,
    size: usize,
    device: Device,
    stream: Stream,
}

impl<T> DevPtr<T>
{

    /// Allocates memory on the device and returns a pointer
    pub fn new(device: Device, size: usize) -> Result<Self, CudaError> {
        let stream = Stream::new();
        Ok( Self { raw_ptr: Box::new(Vec::with_capacity(size)), device, size, stream })
    }

    pub fn bytes(&self) -> usize {
        self.size * std::mem::size_of::<T>()
    }

    pub fn prefetch(&mut self, device: Device) {
    }


    /// Dellocates memory on the device
    fn free(&mut self) {
    }

    /// Copies `count` copies of `value` on the device
    pub fn memset(&mut self, value: u8) {
        unimplemented!()
    }

    pub fn memcpy(&mut self, other: &Self) {
        unimplemented!()
    }

    pub fn as_raw_mut_ptr(&self) -> *mut c_void {
        self.raw_ptr.as_ptr() as *mut c_void
    }

    pub fn as_raw_ptr(&self) -> *const c_void {
        self.raw_ptr.as_ptr() as *mut c_void
    }

    pub fn offset(&self, count: isize) -> Self {
        unimplemented!()
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



