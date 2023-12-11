//#![allow(non_camel_case_types)]

use super::*;

use std::fmt;
use ::std::os::raw::{c_int, c_uint};

use c_uint as cudaError_t;

#[link(name = "cudart")]
extern "C" {
    fn cudaSetDevice(device: c_int) -> cudaError_t;
    fn cudaGetDevice(device: *mut c_int) -> cudaError_t;
    fn cudaGetDeviceCount(count: *mut c_int) -> cudaError_t;
    fn cudaDeviceSynchronize() -> cudaError_t;
}


#[derive(Clone,Copy,PartialEq)]
pub enum Device {
    CPU,
    GPU(i32)
}

impl Device {

  pub fn new(id: i32) -> Self {
    match id {
       -1 => Self::CPU,
       id => Self::GPU(id),
    }
  }

  pub fn id(&self) -> i32 {
    match self {
       Self::CPU => -1,
       Self::GPU(id) => *id,
    }
  }

  pub fn set(&self) {
    let id = self.id();
    wrap_error( (), unsafe { cudaSetDevice(id) } ).unwrap()
  }

  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      match self {
          Device::CPU     => write!(f, "CPU"),
          Device::GPU(id) => write!(f, "GPU({id})")
      }
  }

  pub fn synchronize(&self) {
    self.set();
    wrap_error( (), unsafe { cudaDeviceSynchronize() } ).unwrap()
  }

}

/// Return the current device used for CUDA calls
pub fn get_device() -> Device {
    let mut id: i32 = 0;
    let rc = unsafe { cudaGetDevice(&mut id) };
    wrap_error( Device::new(id), rc ).unwrap()
}

/// Return the number of devices used for CUDA calls
pub fn get_device_count() -> usize {
    let mut id: i32 = 0;
    let rc = unsafe { cudaGetDeviceCount(&mut id) };
    wrap_error( id.try_into().unwrap(), rc ).unwrap()
}


impl fmt::Display for Device {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    self.fmt(f)
  }
}
impl fmt::Debug   for Device {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    self.fmt(f)
  }
}


