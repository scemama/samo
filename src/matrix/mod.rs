use crate::blas_utils;
use core::iter::zip;

use crate::cuda;
use cuda::{Device, DevPtr};

use crate::cublas;

#[derive(Debug,Clone)]
enum Data<T> {
    Rust(Vec::<T>),
    External(*const T),
    ExternalMut(*mut T),
    GPU(DevPtr::<T>),
}

#[derive(Debug,Clone)]
pub struct Matrix<T>
where T: Send + Sync
{
      data: Data<T>,
      lda: usize,
      nrows: usize,
      ncols: usize,
      transposed: bool,
}

unsafe impl<T: Send+Sync> Send for Matrix<T> {}
unsafe impl<T: Send+Sync> Sync for Matrix<T> {}


macro_rules! impl_matrix {
($s:ty) => {

impl Matrix<$s>
{

   #[inline]
   pub fn size(&self) -> usize {
       self.lda * self.ncols
   }

   #[inline]
   pub fn nrows(&self) -> usize {
       if self.transposed { self.ncols } else { self.nrows }
   }

   #[inline]
   pub fn ncols(&self) -> usize {
       if self.transposed { self.nrows } else { self.ncols }
   }

   #[inline]
   pub fn device(&self) -> Device {
       match &self.data {
           Data::<_>::GPU(p) => {
             match p.device() {
                Device::CPU => Device::GPU(0),
                id => id
             }
           },
           _ => Device::CPU,
       }
   }

   pub fn new(device: Device, nrows: usize, ncols: usize) -> Self {
       let size = nrows*ncols;
       let device =
         if cfg!(feature = "cublas") { device } else { Device::CPU };
       let data =
         match device {
           Device::CPU    => Data::<$s>::Rust(vec![ 0. as $s ; size ]),
           Device::GPU(_) => Data::<$s>::GPU( DevPtr::new_managed(device,size).unwrap() )
         };
       let lda = nrows;
       let transposed = false;
       Self { data, lda, nrows, ncols, transposed }
   }

   pub fn from_mut(other: *mut $s, nrows: usize, ncols: usize, lda: usize) -> Self {
       assert!(lda >= nrows);
       let data = Data::<$s>::ExternalMut(other);
       let transposed = false;
       Self { data, lda, nrows, ncols, transposed }
   }

   pub fn from(other: *const $s, nrows: usize, ncols: usize, lda: usize) -> Self {
       assert!(lda >= nrows);
       let data = Data::<$s>::External(other);
       let transposed = false;
       Self { data, lda, nrows, ncols, transposed }
   }

   #[inline]
   pub fn transposed(&self) -> bool {
       self.transposed
   }

   pub fn reshape(&mut self, nrows: usize, ncols: usize) {
       let size = nrows*ncols;

       if self.lda != self.nrows {
           panic!("Can't reshape if leading dimension is not the number of rows: {} {}", self.lda, self.nrows);
       }

       if size != self.size() {
           panic!("New and old sizes don't match: {} {}", size, self.size());
       }

       self.nrows = nrows;
       self.lda   = nrows;
       self.ncols = ncols;
   }

   pub fn copy(&mut self, other: &Self) {
       match &mut self.data {
           Data::<$s>::GPU(v) => { v.memcpy(other.as_slice().as_ptr()); },
           _ => { self.as_slice_mut().copy_from_slice(other.as_slice()); },
       }
   }

   /// Creates a new matrix using a submatrix of the current matrix. The data is shared.
   pub fn submatrix(&mut self, init_rows: usize, init_cols: usize, nrows: usize, ncols: usize) -> Self {
      assert!(ncols <= self.ncols);
      assert!(nrows <= self.lda);
      assert!(init_cols <= self.ncols);
      assert!(init_rows <= self.nrows);

      let lda = self.lda;
      let transposed = self.transposed;
      let offset: isize = (init_rows + lda*init_cols) as isize;
      let data = match &mut self.data {
         Data::<$s>::Rust(v) => unsafe { Data::<$s>::ExternalMut(v.as_mut_ptr().offset(offset)) },
         Data::<$s>::External(v) => unsafe { Data::<$s>::External((*v as *const $s).offset(offset)) },
         Data::<$s>::ExternalMut(v) => unsafe { Data::<$s>::ExternalMut((*v as *mut $s).offset(offset)) },
         Data::<$s>::GPU(v) => Data::<$s>::GPU(v.offset(offset)),
      };
      Self { data, lda, nrows, ncols, transposed }
   }

   pub fn as_slice(&self) -> &[$s] {
       match &self.data {
           Data::<$s>::Rust(v) => &v[..],
           Data::<$s>::ExternalMut(v) => unsafe {std::slice::from_raw_parts(*v, self.size()) },
           Data::<$s>::External(v) => unsafe {std::slice::from_raw_parts(*v as *mut $s, self.size()) },
           Data::<$s>::GPU(v) => v.as_slice(),
       }
   }

   pub fn as_slice_mut(&mut self) -> &mut [$s] {
       let size = self.size();
       match &mut self.data {
           Data::<$s>::Rust(v) => &mut v[..],
           Data::<$s>::ExternalMut(v) => unsafe {std::slice::from_raw_parts_mut(*v, size) },
           Data::<$s>::External(_) => panic!("Immutable matrix"),
           Data::<$s>::GPU(v) => v.as_slice_mut(),
       }
   }

   pub fn t(&self) -> Self {
       let data = match &self.data {
           Data::<$s>::Rust(v) => Data::<$s>::External(v.as_ptr()),
           Data::<$s>::ExternalMut(v) => Data::<$s>::External(*v as *const $s),
           Data::<$s>::External(v) => Data::<$s>::External(*v),
           Data::<$s>::GPU(v) => Data::<$s>::GPU(v.clone()),
       };
       Self { transposed: !self.transposed,
              data, ..*self}
   }

   pub fn t_mut(&mut self) {
       self.transposed = !self.transposed;
   }

   pub fn prefetch_to(&self, device: Device) {
       match &self.data {
          Data::<$s>::GPU(v) => v.prefetch_to(device),
          _ => (),
       }
   }

} // end impl Matrix

impl std::ops::Index<[ usize ; 2 ]> for Matrix<$s>
{
    type Output = $s;

    #[inline]
    fn index(&self, idx: [ usize; 2 ] ) -> &Self::Output {
        let i = idx[0];
        let j = idx[1];
        let data = self.as_slice();
        let lda = self.lda;
        match self.transposed {
            false => { assert!(i < self.nrows && j < self.ncols); &data[i+j*lda] },
            true  => { assert!(j < self.nrows && i < self.ncols); &data[j+i*lda] },
        }
    }
}

impl std::ops::IndexMut<[ usize ; 2 ]> for Matrix<$s>
{
    #[inline]
    fn index_mut(&mut self, idx: [usize ; 2]) -> &mut Self::Output {
        let i = idx[0];
        let j = idx[1];
        let transposed = self.transposed;
        let nrows = self.nrows;
        let ncols = self.ncols;
        let lda = self.lda;
        let data = self.as_slice_mut();
        match transposed {
            false => {assert!(i < nrows && j < ncols); &mut data[i+j*lda]},
            true  => {assert!(j < nrows && i < ncols); &mut data[j+i*lda]},
        }
    }
}

}} // end macro

impl_matrix!(f32);
impl_matrix!(f64);

//--------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creation() {
        let m = 11;
        let n = 12;
        for device in [ Device::CPU, Device::GPU(0) ] {
            let matrix = Matrix::<f64>::new(Device::CPU, m, n);
            assert_eq!(matrix.nrows, m);
            assert_eq!(matrix.ncols, n);
            for j in 0..(matrix.ncols) {
                for i in 0..(matrix.nrows) {
                    assert_eq!(matrix[[i,j]], 0.0);
                }
            }

            let mut a = vec![ 0. ; m*n ];
            for j in 0..n {
                for i in 0..m {
                    a[i + j*m] = (i as f64) + (j as f64)*1000.0;
                }
            }
            let matrix = Matrix::<f64>::from(a.as_mut_ptr(), m, n, m);
            for j in 0..(matrix.ncols) {
                for i in 0..(matrix.nrows) {
                    assert_eq!(matrix[[i,j]], a[i+j*m]);
                }
            }
        }
    }

    #[test]
    fn reshape() {
        let m = 20;
        let n = 30;
        let mut a = vec![ 0. ; m*n ];
        let mut a_ref = vec![ 0. ; m*n ];
        for j in 0..n {
            for i in 0..m {
                a[i + j*m] = (i as f64) + (j as f64)*1000.0;
                a_ref[i + j*m] = (i as f64) + (j as f64)*1000.0;
            }
        }
        let mut a_mat = Matrix::<f64>::from(a.as_mut_ptr(), m, n, m);
        a_mat.reshape(60,10);
        assert_eq!(a_mat.nrows, 60);
        assert_eq!(a_mat.ncols, 10);
        let a = a_mat.as_slice();
        assert_eq!(a_ref, a);
    }

    #[test]
    fn transposition() {
        let m = 21;
        let n = 22;
        let mut a = vec![ 0. ; m*n ];
        let mut a_t = vec![ 0. ; m*n ];
        for j in 0..n {
            for i in 0..m {
                a  [i + j*m] = (i as f64) + (j as f64)*1000.0;
                a_t[j + i*n] = (i as f64) + (j as f64)*1000.0;
            }
        }
        let a = Matrix::<f64>::from(a.as_mut_ptr(), m, n, m);
        let a_t = Matrix::<f64>::from(a_t.as_mut_ptr(), n, m, n);

        let b = a_t.t();
        assert!(!a.transposed());
        assert!(!a_t.transposed());
        assert!(b.transposed());
        for j in 0..n {
            for i in 0..m {
                assert_eq!(a[[i,j]], a_t[[j,i]]);
                assert_eq!(a[[i,j]], b[[i,j]]);
            }
        }
    }

    #[test]
    #[should_panic]
    fn geam_wrong_size() {
        let n = 10;
        let m = 5;
        let a = Matrix::<f64>::new(Device::CPU, m, n);
        let b = Matrix::<f64>::new(Device::CPU, n, m);
        let _ = Matrix::<f64>::geam(None, 1.0, &a, 1.0, &b);
    }
}

mod gemm; pub use gemm::*;
mod geam; pub use geam::*;
