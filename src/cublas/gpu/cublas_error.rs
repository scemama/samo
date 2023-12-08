use super::*;
use std::{fmt, error};

pub type cublasStatus_t = ::std::os::raw::c_uint;
pub struct CublasError(pub cublasStatus_t);

pub const SUCCESS: cublasStatus_t = 0;
pub const NOT_INITIALIZED: cublasStatus_t = 1;
pub const ALLOC_FAILED: cublasStatus_t = 3;
pub const INVALID_VALUE: cublasStatus_t = 7;
pub const ARCH_MISMATCH: cublasStatus_t = 8;
pub const MAPPING_ERROR: cublasStatus_t = 11;
pub const EXECUTION_FAILED: cublasStatus_t = 13;
pub const INTERNAL_ERROR: cublasStatus_t = 14;
pub const NOT_SUPPORTED: cublasStatus_t = 15;
pub const LICENSE_ERROR: cublasStatus_t = 16;

fn fmt_error(s: &CublasError, f: &mut fmt::Formatter<'_>) -> fmt::Result {
       match s.0 {
        SUCCESS           =>  write!(f, "Success"),
        NOT_INITIALIZED   =>  write!(f, "Not initialized"),
        ALLOC_FAILED      =>  write!(f, "Allocation failed"),
        INVALID_VALUE     =>  write!(f, "Invalid value"),
        ARCH_MISMATCH     =>  write!(f, "Arch mismatch"),
        MAPPING_ERROR     =>  write!(f, "Mapping error"),
        EXECUTION_FAILED  =>  write!(f, "Execution failed"),
        INTERNAL_ERROR    =>  write!(f, "Interal error"),
        NOT_SUPPORTED     =>  write!(f, "Not supported"),
        LICENSE_ERROR     =>  write!(f, "License error"),
        i  => write!(f, "Error {i}"),
       }
}

impl fmt::Display for CublasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_error(self,f)
    }
}

impl fmt::Debug for CublasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_error(self,f)
    }
}

impl error::Error for CublasError {}

pub fn wrap_error<T>(output: T, e: cublasStatus_t) -> Result<T, CublasError> {
    match e {
        0 => Ok(output),
        _ => Err(CublasError(e)),
    }
}

