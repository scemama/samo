use super::*;
use crate::cuda;
use crate::cuda::cudaStream_t;

extern "C" {
    pub fn cublasSetStream_v2(handle: cublasHandle_t, streamId: cudaStream_t) -> cublasStatus_t;
    pub fn cublasGetStream_v2(
        handle: cublasHandle_t,
        streamId: *mut cudaStream_t,
    ) -> cublasStatus_t;
}

impl cuda::Stream {
    pub fn set_active(&self, handle: &Context) -> Result<(), CublasError> {
      let status = unsafe {
        cublasSetStream_v2(handle.as_cublasHandle_t(), self.as_cudaStream_t())
      };
      wrap_error( (), status)
    }

    pub fn release(&self, handle: &Context) -> Result<(), CublasError> {
      let null = std::ptr::null_mut();
      let status = unsafe {
        cublasSetStream_v2(handle.as_cublasHandle_t(), null)
      };
      wrap_error( (), status)
    }
}

