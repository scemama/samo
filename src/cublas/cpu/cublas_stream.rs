use super::*;
use crate::cuda;

impl cuda::Stream {
    pub fn set_active(&self, handle: &Context) -> Result<(), CublasError> {
      Ok( () )
    }

    pub fn release(&self, handle: &Context) -> Result<(), CublasError> {
      Ok( () )
    }
}

