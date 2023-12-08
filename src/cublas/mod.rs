#[cfg(feature="cublas")]
pub mod gpu;
#[cfg(feature="cublas")]
pub use gpu::*;

#[cfg(not(feature="cublas"))]
pub mod cpu;
#[cfg(not(feature="cublas"))]
pub use cpu::*;
