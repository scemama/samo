use super::*;

use std::fmt;
use ::std::os::raw::{c_void};
use std::cell::Cell;

pub struct MemInfo {
    pub free: usize,
    pub total: usize
}

/// Returns the amount of free and total memory on the device
pub fn get_mem_info() -> Result<MemInfo, CudaError> {
  let free = 1000000000;
  let total = 1000000000;
  Ok( MemInfo {free, total} )
}


/// Pointer to memory on the device
#[derive(Clone)]
pub struct DevPtr<T> {
    raw_ptr: Box::<Vec<T>>,
    size: usize,
    device: Cell<Device>,
    stream: Stream,
}

impl<T> DevPtr<T>
{

    #[inline]
    pub fn device(&self) -> Device {
        self.device.get()
    }

    /// Allocates memory on the device and returns a pointer
    pub fn new(device: Device, size: usize) -> Result<Self, CudaError> {
        let stream = Stream::new();
        Ok( Self { raw_ptr: Box::new(Vec::with_capacity(size)), device: Cell::new(device), size, stream })
    }

    pub fn new_managed(device: Device, size: usize) -> Result<Self, CudaError> {
        let stream = Stream::new();
        Ok( Self { raw_ptr: Box::new(Vec::with_capacity(size)), device: Cell::new(device), size, stream })
    }

    pub fn bytes(&self) -> usize {
        self.size * std::mem::size_of::<T>()
    }

    pub fn prefetch_to(&self, _: Device) {
    }

    pub fn prefetch_only(&self, _: usize) {
    }

    pub fn prefetch_columns(&self, _: usize, _: usize, _: usize) {
    }

    pub fn prefetch(&mut self, _: Device) {
    }



    /// Copies `count` copies of `value` on the device
    pub fn memset(&mut self, value: u8) {
        unimplemented!()
    }

    pub fn memcpy(&mut self, other: *const T) {
        unimplemented!();
        /*
        let other = std::slice::from_raw_parts(other, self.size);
        for i in 0..self.size {
            self.raw_ptr[i] = other[i];
        }
        */
    }

    pub fn as_raw_mut_ptr(&self) -> *mut c_void {
        self.raw_ptr.as_ptr() as *mut c_void
    }

    pub fn as_raw_ptr(&self) -> *const c_void {
        self.raw_ptr.as_ptr() as *const c_void
    }

    pub fn as_slice_mut(&self) -> &mut [T] {
         unsafe { std::slice::from_raw_parts_mut(self.as_raw_mut_ptr() as *mut T, self.size) }
    }

    pub fn as_slice(&self) -> &[T] {
         unsafe { std::slice::from_raw_parts(self.as_raw_ptr() as *const T, self.size) }
    }

    pub fn offset(&self, count: isize) -> Self {
        unimplemented!();
        /*
        let offset: isize = count;
        let new_size: usize = ( (self.size as isize) - count).try_into().unwrap();
        let raw_ptr = unsafe { self.raw_ptr.as_ptr().offset(offset) };
        let stream = Stream::new();
        Self { raw_ptr: Box::new(self.raw_ptr[(offset as usize)..].to_vec()), device: Cell::new(self.device.get()), stream, size: new_size }
        */
    }

#[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{ptr: {:?}, device: {}, size: {}}}",
        self.raw_ptr.as_ptr(),
        self.device.get(),
        self.size)
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



