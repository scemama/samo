//#![allow(non_camel_case_types)]

use super::*;

use std::fmt;
use ::std::os::raw::{c_int, c_uint};

use c_uint as cudaError_t;


#[derive(Clone,Copy)]
pub enum Device {
    CPU,
    GPU(i32)
}

impl Device {

  pub fn new(id: i32) -> Self {
    match id {
       -1 => Self::CPU,
       id => panic!("cublas feature disabled"),
    }
  }

  pub fn id(&self) -> i32 {
    match self {
       Self::CPU => -1,
       Self::GPU(id) => *id,
    }
  }

  pub fn set(&self) {
    ()
  }

  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      match self {
          Device::CPU     => write!(f, "CPU"),
          Device::GPU(id) => write!(f, "GPU({id})")
      }
  }

  pub fn synchronize(&self) {
    ()
  }

}

/// Return the current device used for CUDA calls
pub fn get_device() -> Device {
    Device::CPU
}

/// Return the number of devices used for CUDA calls
pub fn get_device_count() -> usize {
    0
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


