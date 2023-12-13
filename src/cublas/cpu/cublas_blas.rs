use super::*;
use crate::cuda::DevPtr;

pub fn dgemm (handle: &Context,
             transa: u8,
             transb: u8,
             m: usize,
             n: usize,
             k: usize,
             alpha: f64,
             a: &DevPtr<f64>,
             lda: usize,
             b: &DevPtr<f64>,
             ldb: usize,
             beta: f64,
             c: &mut DevPtr<f64>,
             ldc: usize
            ) -> Result<(), CublasError>
{
    unimplemented!();
    Ok( () )
}


pub fn sgemm (handle: &Context,
             transa: u8,
             transb: u8,
             m: usize,
             n: usize,
             k: usize,
             alpha: f32,
             a: &DevPtr<f32>,
             lda: usize,
             b: &DevPtr<f32>,
             ldb: usize,
             beta: f32,
             c: &mut DevPtr<f32>,
             ldc: usize
            ) -> Result<(), CublasError>
{
    Ok( () )
}


pub fn dgeam (handle: &Context,
             transa: u8,
             transb: u8,
             m: usize,
             n: usize,
             alpha: f64,
             a: &DevPtr<f64>,
             lda: usize,
             beta: f64,
             b: &DevPtr<f64>,
             ldb: usize,
             c: &mut DevPtr<f64>,
             ldc: usize
            ) -> Result<(), CublasError>
{
    unimplemented!();
    Ok( () )
}

pub fn sgeam (handle: &Context,
             transa: u8,
             transb: u8,
             m: usize,
             n: usize,
             alpha: f32,
             a: &DevPtr<f32>,
             lda: usize,
             beta: f32,
             b: &DevPtr<f32>,
             ldb: usize,
             c: &mut DevPtr<f32>,
             ldc: usize
            ) -> Result<(), CublasError>
{
    unimplemented!();
    Ok( () )
}

