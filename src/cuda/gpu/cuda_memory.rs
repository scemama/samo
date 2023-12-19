use super::*;

use std::sync::Arc;
use std::ptr::NonNull;
use std::fmt;
use ::std::os::raw::{c_void, c_int, c_uint};
use std::marker::PhantomData;
use std::cell::Cell;

use c_uint as cudaError_t;

pub enum MemAdvise {
  SetReadMostly,
  UnsetReadMostly,
  SetPreferredLocation,
  UnsetPreferredLocation,
  SetAccessedBy,
  UnsetAccessedBy,
}

impl From<u32> for MemAdvise {
  fn from(i: u32) -> MemAdvise {
    match i {
      1 => MemAdvise::SetReadMostly,
      2 => MemAdvise::UnsetReadMostly,
      3 => MemAdvise::SetPreferredLocation,
      4 => MemAdvise::UnsetPreferredLocation,
      5 => MemAdvise::SetAccessedBy,
      6 => MemAdvise::UnsetAccessedBy,
      _ => panic!("Unknown value")
    }
  }
}

impl Into<u32> for MemAdvise {
  fn into(self: MemAdvise) -> u32 {
    match self {
      MemAdvise::SetReadMostly => 1,
      MemAdvise::UnsetReadMostly => 2,
      MemAdvise::SetPreferredLocation => 3,
      MemAdvise::UnsetPreferredLocation => 4,
      MemAdvise::SetAccessedBy => 5,
      MemAdvise::UnsetAccessedBy => 6,
    }
  }
}

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
    fn cudaMemAdvise (
        devPtr: *const c_void,
        count: usize,
        advice: c_uint,
        device: c_int
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
struct CudaDevPtr {
   ptr: NonNull<c_void>,
   original: bool,
}

impl CudaDevPtr {
  fn as_ptr(&self) -> *mut c_void {
    self.ptr.as_ptr()
  }
}

impl Drop for CudaDevPtr {
    fn drop(&mut self) {
        if self.original {
          wrap_error( (), unsafe { cudaFree(self.ptr.as_ptr()) } ).unwrap()
        }
    }
}

#[derive(Clone)]
pub struct DevPtr<T> {
    raw_ptr: Arc<CudaDevPtr>,
    size: usize,
    device: Cell<Device>,
    stream: Stream,
    _phantom: PhantomData<T>,
}

impl<T> DevPtr<T>
{

    #[inline]
    pub fn device(&self) -> Device {
        self.device.get()
    }

    /// Allocates memory on the device and returns a pointer
    pub fn new(device: Device, size: usize) -> Result<Self, CudaError> {
        device.activate();
        let mut raw_ptr = std::ptr::null_mut();
        let stream = Stream::new();
        let rc = unsafe { cudaMalloc(&mut raw_ptr as *mut *mut c_void,
                                     size * std::mem::size_of::<T>()  ) };
        NonNull::new(raw_ptr).map(|raw_ptr|
        Self { raw_ptr: Arc::new(CudaDevPtr {ptr: raw_ptr, original:true}), device: Cell::new(device), size, stream, _phantom: PhantomData })
           .ok_or(CudaError(rc))
    }

    /// Allocates memory on the device and returns a pointer
    pub fn new_managed(device: Device, size: usize) -> Result<Self, CudaError> {
        device.activate();
        let mut raw_ptr = std::ptr::null_mut();
        let stream = Stream::new();
        let rc = unsafe { cudaMallocManaged(&mut raw_ptr as *mut *mut c_void,
                                     size * std::mem::size_of::<T>(), 1  ) };
        NonNull::new(raw_ptr).map(|raw_ptr|
        { let r = Self { raw_ptr: Arc::new(CudaDevPtr {ptr: raw_ptr, original:true}), device: Cell::new(device), size, stream, _phantom: PhantomData };
        r.mem_advise(MemAdvise::SetAccessedBy); r }
        ).ok_or(CudaError(rc))
    }

    pub fn mem_advise(&self, advice: MemAdvise) {
      let advice : u32 = advice.into();
      wrap_error( (), unsafe {
        cudaMemAdvise(self.raw_ptr.as_ptr(), self.bytes(), advice as c_uint,
          self.device.get().id())} ).unwrap()
    }

    pub fn bytes(&self) -> usize {
        self.size * std::mem::size_of::<T>()
    }

    pub fn prefetch_to(&self, device: Device) {
        self.device.set(device);
        wrap_error( (), unsafe {
            cudaMemPrefetchAsync(self.raw_ptr.as_ptr(), self.bytes(), device.id(),
                self.stream.as_cudaStream_t()) }).unwrap()
    }

    pub fn prefetch_only(&self, count: usize) {
        wrap_error( (), unsafe {
            cudaMemPrefetchAsync(self.raw_ptr.as_ptr(), count * std::mem::size_of::<T>(), self.device.get().id(),
                self.stream.as_cudaStream_t()) }).unwrap()
    }

    pub fn prefetch_columns(&self, count: usize, lda: usize, ncolumns: usize) {
        wrap_error( (), unsafe {
            let id = self.device.get().id();
            let stride = lda*std::mem::size_of::<T>();
            let mut ptr = self.raw_ptr.as_ptr();
            let bytes = count * std::mem::size_of::<T>();
            let stream = self.stream.as_cudaStream_t();
            let mut rc = 0;
            for i in 0..ncolumns {
                rc = cudaMemPrefetchAsync(ptr, bytes, id, stream);
                let ptr = ptr.offset(stride as isize);
                if rc != 0 { break };
            }
            rc }).unwrap()
    }

    /// Copies `count` copies of `value` on the device
    pub fn memset(&mut self, value: u8) {
        self.device.get().activate();
        wrap_error( (), unsafe {
            cudaMemset(self.raw_ptr.as_ptr(), value as c_int, self.bytes())
            } ).unwrap()
    }

    pub fn memcpy(&mut self, other: *const T) {
        self.device.get().activate();
        wrap_error( (), unsafe {
            cudaMemcpy(self.raw_ptr.as_ptr(),
                       other as *const c_void,
                       self.bytes(), 4)
            } ).unwrap()
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
        let offset: isize = count * (std::mem::size_of::<T>() as isize);
        let new_size: usize = ( (self.size as isize) - count).try_into().unwrap();
        let raw_ptr = unsafe { self.raw_ptr.as_ptr().offset(offset) };
        let stream = Stream::new();
        NonNull::new(raw_ptr).map(|raw_ptr|
           Self { raw_ptr: Arc::new(CudaDevPtr {ptr: raw_ptr, original: false}), device: Cell::new(self.device.get()), stream, size: new_size, _phantom: PhantomData, }).unwrap()
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



// ------------------------------------------------------------------------

mod tests {

    use super::*;

    #[test]
    fn memory() {
        let info = get_mem_info().unwrap();
        println!("Free: {}\nTotal: {}", info.free, info.total);
        assert!(info.free > 0);
        assert!(info.total > 0);

        let mut dev_ptr = DevPtr::<f64>::new(Device::GPU(0),10).unwrap();
        dev_ptr.memset(1);
    }
}

