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


macro_rules! impl_matrix {
($s:ty, $gemm_cpu:path, $gemm_gpu:path) => {

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

   //------

   pub fn gemm(alpha: $s, a: &Self, b: &Self) -> Self {
       let device =
           match (a.device(), b.device()) {
                (Device::GPU(d), Device::GPU(_)) => Device::GPU(d),
                _ => Device::CPU,
            };
       let mut c = Self::new(device, a.nrows(), b.ncols());
       Self::gemm_mut(alpha, a, b, 0.0, &mut c);
       c
   }

   fn recursive_gemm_nn(handle: &cublas::Context, m:usize, n:usize, k:usize, alpha: $s,
              a_ptr: &DevPtr<$s>, lda:usize,
              b_ptr: &DevPtr<$s>, ldb:usize, beta: $s,
              c_ptr: &mut DevPtr<$s>, ldc:usize, block_size: usize) {

      if (ldb*n > block_size || ldc*n > block_size) {

        let n1 = n/2;
        let n2 = n - n1;
        let b_ptr2 = b_ptr.offset((ldb*n1) as isize);
        let mut c_ptr2 = c_ptr.offset((ldc*n1) as isize);

        Self::recursive_gemm_nn(handle, m, n1, k,
                  alpha, &a_ptr, lda, &b_ptr, ldb, beta,
                  c_ptr, ldc, block_size);

        Self::recursive_gemm_nn(handle, m, n2, k,
                  alpha, &a_ptr, lda, &b_ptr2, ldb, beta,
                  &mut c_ptr2, ldc, block_size);

      } else if (m*n > block_size) {

        let m1 = m/2;
        let m2 = m - m1;
        let a_ptr2 = a_ptr.offset(m1 as isize);
        let mut c_ptr2 = c_ptr.offset(m1 as isize);

        Self::recursive_gemm_nn(handle, m1, n, k,
                  alpha, &a_ptr, lda, &b_ptr, ldb, beta,
                  c_ptr, ldc, block_size);

        Self::recursive_gemm_nn(handle, m2, n, k,
                  alpha, &a_ptr2, lda, &b_ptr, ldb, beta,
                  &mut c_ptr2, ldc, block_size);

      } else {


       c_ptr.prefetch_only(ldc * (n - 1) + m);

       let chunk_size = block_size / lda;
       let n_chunks = k/chunk_size+1 ;

       for i in 0..n_chunks {

            let offset = i * chunk_size;
            let current_k = if i < n_chunks - 1 { chunk_size } else { k - offset };

            let a_offset_ptr = a_ptr.offset((lda * offset) as isize);
            let b_offset_ptr = b_ptr.offset(offset as isize);

            a_offset_ptr.prefetch_only(lda * (current_k-1) + m );
            b_offset_ptr.prefetch_only(ldb * (n-1) + current_k);

            $gemm_gpu(handle, b'N', b'N', m, n, current_k, alpha,
                      &a_offset_ptr, lda, &b_offset_ptr, ldb,
                      if i == 0 { beta } else { 1.0 }, c_ptr, ldc).unwrap();
            }

       }
   }

   pub fn gemm_mut(alpha: $s, a: &Self, b: &Self, beta: $s, c: &mut Self)
   {
       if c.transposed {
           panic!("Can't write in a transposed matrix");
       }

       if a.ncols() != b.nrows() {
           panic!("a.ncols() != b.nrows() : {} {}", a.ncols(), b.nrows());
       }

       if a.nrows() != c.nrows() {
           panic!("a.nrows() != c.nrows() : {} {}", a.nrows(), c.nrows());
       }

       if b.ncols() != c.ncols() {
           panic!("b.ncols() != c.ncols() : {} {}", b.ncols(), c.ncols());
       }

       let transa = if a.transposed { b'T' } else { b'N' };
       let transb = if b.transposed { b'T' } else { b'N' };

       let ldc = c.lda;

       match (&a.data, &b.data, &mut c.data) {
         (Data::<$s>::GPU(a_ptr), Data::<$s>::GPU(b_ptr), Data::<$s>::GPU(c_ptr)) => {
                // Run on GPU
            let handle = cublas::Context::new().unwrap();
            let mem = cuda::get_mem_info().unwrap();
            let block_size = mem.total / (8*8);
            let lda = a.lda;
            let ldb = b.lda;
            let ldc = c.lda;
            match (a.transposed, b.transposed) {
                (false, false) => Self::recursive_gemm_nn(&handle, c.nrows, c.ncols, b.nrows(),
                                      alpha, a_ptr, a.lda, b_ptr, b.lda, beta,
                                      c_ptr, ldc, block_size),
                (true, false) => unimplemented!(),
                (false, true) => unimplemented!(),
                (true, true ) => unimplemented!(),
            };
          },

         _ => {  // Run on CPU
            $gemm_cpu(transa, transb, c.nrows, c.ncols, b.nrows(),
                    alpha, a.as_slice(), a.lda, b.as_slice(), b.lda, beta,
                    c.as_slice_mut(), ldc)
          },

       };
   }

   pub fn geam_mut(alpha: $s, a: &Self, beta: $s, b: &Self, c: &mut Self)
   {
       if c.transposed {
           panic!("Can't write in a transposed matrix");
       }

       if a.ncols() != b.ncols() {
           panic!("a.ncols() != b.ncols() : {} {}", a.ncols(), b.ncols());
       }

       if a.ncols() != c.ncols() {
           panic!("a.ncols() != c.ncols() : {} {}", a.ncols(), c.ncols());
       }

       if a.nrows() != b.nrows() {
           panic!("a.nrows() != b.nrows() : {} {}", a.nrows(), b.nrows());
       }

       if a.nrows() != c.nrows() {
           panic!("a.nrows() != c.nrows() : {} {}", a.nrows(), c.nrows());
       }

       let nrows = a.nrows;
       let ncols = a.ncols;
       let mut transposed = false;
       let make_pattern = |x| {
           if x == 0.0 { 0 }
           else if x == 1.0 { 1 }
           else if x == -1.0 { -1 }
           else { 2 }
       };

       let _a = make_pattern(alpha);
       let _b = make_pattern(beta);

       match (_a, _b, a.transposed, b.transposed) {
           (0,0,_,_) =>  {
               for x in c.as_slice_mut() {
                   *x = 0.0;
               }
           },

           (1,0,_,_) =>  {
               transposed = a.transposed;
               for (x, &v) in zip(c.as_slice_mut(), a.as_slice()) {
                   *x = v;
               }
           },

           (-1,0,_,_) =>  {
               transposed = a.transposed;
               for (x, &v) in zip(c.as_slice_mut(), a.as_slice()) {
                   *x = -v;
               }
           },

           (_,0,_,_) =>  {
               transposed = a.transposed;
               for (x, &v) in zip(c.as_slice_mut(), a.as_slice()) {
                   *x = alpha*v;
               }
           },

           (0,1,_,_) =>  {
               transposed = b.transposed;
               for (x, &v) in zip(c.as_slice_mut(), b.as_slice()) {
                   *x = v;
               }
           },

           (0,-1,_,_) =>  {
               transposed = b.transposed;
               for (x, &v) in zip(c.as_slice_mut(), b.as_slice()) {
                   *x = -v;
               }
           },

           (0,_,_,_) =>  {
               transposed = b.transposed;
               for (x, &v) in zip(c.as_slice_mut(), b.as_slice()) {
                   *x = beta*v;
               }
           },

           (1, 1, false, false) | (1, 1, true, true) => {
               transposed = a.transposed;
               for (x, (&v, &w)) in zip(c.as_slice_mut(), zip(a.as_slice(), b.as_slice())) {
                   *x = v + w;
               }},

           (1,-1, false, false) | (1,-1, true, true) => {
               transposed = a.transposed;
               for (x, (&v, &w)) in zip(c.as_slice_mut(), zip(a.as_slice(), b.as_slice())) {
                   *x = v - w;
               }},

           (_, _, false, false) | (_, _, true, true) => {
               transposed = a.transposed;
               for (x, (&v, &w)) in zip(c.as_slice_mut(), zip(a.as_slice(), b.as_slice())) {
                   *x = alpha * v + beta * w;
               }},

           (_, _, true, false) => {
               transposed = a.transposed;
               let a_ = a.as_slice();
               let b_ = b.as_slice();
               let ldc = c.lda;
               let c_ = c.as_slice_mut();
               for i in 0..ncols   {
                   for j in 0..nrows   {
                       let x = a_[j+i*a.lda];
                       let y = b_[i+j*b.lda];
                       c_[j+i*ldc] = alpha * x + beta * y ;
                   }
               } },

           (_, _, false, true) => {
               transposed = a.transposed;
               let a_ = a.as_slice();
               let b_ = b.as_slice();
               let ldc = c.lda;
               let c_ = c.as_slice_mut();
               for j in 0..ncols   {
                   for i in 0..nrows   {
                   let x = a_[i+j*a.lda];
                   let y = b_[j+i*b.lda];
                   c_[i+j*ldc] = alpha * x + beta * y ;
                   }
               } },
       };
       c.transposed = transposed;
   }

   pub fn geam(alpha: $s, a: &Self, beta: $s, b: &Self) -> Self {
       let device =
           match (a.device(), b.device()) {
                (Device::GPU(d), Device::GPU(_)) => Device::GPU(d),
                _ => Device::CPU,
            };
       let mut c = Self::new(device, a.nrows(), b.ncols());
       Self::geam_mut(alpha, a, beta, b, &mut c);
       c
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

impl_matrix!(f32, blas_utils::sgemm, cublas::sgemm);
impl_matrix!(f64, blas_utils::dgemm, cublas::dgemm);

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
    fn dgemm() {
        let mut a_vec = vec![1. , 1.1, 1.2, 1.3,
                         2. , 2.1, 2.2, 2.3,
                         3. , 3.1, 3.2, 3.3];

        let mut b_vec = vec![ 1.0, 2.0, 3.0, 4.0,
                          1.1, 2.1, 3.1, 4.1];

        let mut c_vec = vec![12.0 , 22.0 , 32.0,
                         12.46, 22.86, 33.26];

        let a = Matrix::<f64>::from( a_vec.as_mut_ptr(), 4, 3, 4).t();
        let b = Matrix::<f64>::from( b_vec.as_mut_ptr(), 4, 2, 4 );
        let c_ref = Matrix::<f64>::from( c_vec.as_mut_ptr(), 3, 2, 3);

        let c = Matrix::<f64>::gemm(1.0, &a, &b);

        let difference = Matrix::<f64>::geam(1.0, &c, -1.0, &c_ref);
        for j in 0..2 {
            for i in 0..3 {
                assert!(num::abs(difference[[i,j]] / c[[i,j]]) < <f64>::EPSILON);
            }
        }

        let a = a.t();
        let b = b.t();
        let c_t = Matrix::<f64>::gemm(1.0, &b, &a);
        let c_t2 = c.t();
        for j in 0..3 {
            for i in 0..2 {
                assert_eq!(c_t[[i,j]], c_t2[[i,j]]);
            }
        }



        let a_gpu = Matrix::<f64>::new(Device::GPU(0), 4, 3).t();
        let b_gpu = Matrix::<f64>::new(Device::GPU(0), 4, 2);
        let c_gpu = Matrix::<f64>::gemm(1.0, &a_gpu, &b_gpu);

        let difference = Matrix::<f64>::geam(1.0, &c, -1.0, &c_ref);
        for j in 0..2 {
            for i in 0..3 {
                assert!(num::abs(difference[[i,j]] / c[[i,j]]) < <f64>::EPSILON);
            }
        }

        let a_gpu = a_gpu.t();
        let b_gpu = b_gpu.t();
        let c_t = Matrix::<f64>::gemm(1.0, &b_gpu, &a_gpu);
        let c_t2 = c_gpu.t();
        for j in 0..3 {
            for i in 0..2 {
                assert_eq!(c_t[[i,j]], c_t2[[i,j]]);
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
        let _ = Matrix::<f64>::geam(1.0, &a, 1.0, &b);
    }

    #[test]
    fn geam_cases() {
        let n = 8;
        let m = 4;
        let mut a   = Matrix::<f64>::new(Device::CPU, m, n);
        let mut a_t = Matrix::<f64>::new(Device::CPU, n, m);
        let mut b   = Matrix::<f64>::new(Device::CPU, m, n);
        let mut b_t = Matrix::<f64>::new(Device::CPU, n, m);
        let zero_mat = Matrix::<f64>::new(Device::CPU, m, n);
        let zero_mat_t = Matrix::<f64>::new(Device::CPU, n, m);
        for i in 0..m {
            for j in 0..n {
                a[[i,j]] = (i * 10 + j) as f64;
                b[[i,j]] = (i * 1000 + j*10) as f64;
                a_t[[j,i]] = (i * 10 + j) as f64;
                b_t[[j,i]] = (i * 1000 + j*10) as f64;
            }
        }

        assert_eq!(
            Matrix::<f64>::geam(0.0, &a, 0.0, &b).as_slice(),
            zero_mat.as_slice());

        assert_eq!(
            Matrix::<f64>::geam(1.0, &a, 0.0, &b).as_slice(),
            a.as_slice());

        assert_eq!(
            Matrix::<f64>::geam(0.0, &a, 1.0, &b).as_slice(),
            b.as_slice());

        let mut r = Matrix::<f64>::new(Device::CPU, m, n);
        for i in 0..m {
            for j in 0..n {
                r[[i,j]] = 2.0*a[[i,j]];
            }
        };
        assert_eq!(
            Matrix::<f64>::geam(2.0, &a, 0.0, &b).as_slice(),
            r.as_slice());

        assert_eq!(
            Matrix::<f64>::geam(0.5, &a, 0.5, &a).as_slice(),
            a.as_slice());

        assert_eq!(
            Matrix::<f64>::geam(0.5, &a, -0.5, &a).as_slice(),
            zero_mat.as_slice());
        assert_eq!(
            Matrix::<f64>::geam(1.0, &a, -1.0, &a).as_slice(),
            zero_mat.as_slice());

        let mut r = Matrix::<f64>::new(Device::CPU, m, n);
        for i in 0..m {
            for j in 0..n {
                r[[i,j]] = 2.0*b[[i,j]];
            }
        };
        println!("{:?}", a);
        println!("{:?}", b);
        assert_eq!(
            Matrix::<f64>::geam(0.0, &a, 2.0, &b).as_slice(),
            r.as_slice() );

        let mut r = Matrix::<f64>::new(Device::CPU, m, n);
        for i in 0..m {
            for j in 0..n {
                r[[i,j]] = a[[i,j]] + b[[i,j]];
            }
        };
        assert_eq!(
            Matrix::<f64>::geam(1.0, &a, 1.0, &b).as_slice(),
            r.as_slice());

        let mut r = Matrix::<f64>::new(Device::CPU, m, n);
        for i in 0..m {
            for j in 0..n {
                r[[i,j]] = b[[i,j]] - a[[i,j]];
            }
        };
        assert_eq!(
            Matrix::<f64>::geam(-1.0, &a, 1.0, &b).as_slice(),
            r.as_slice());

        let mut r = Matrix::<f64>::new(Device::CPU, m, n);
        for i in 0..m {
            for j in 0..n {
                r[[i,j]] = a[[i,j]] - b[[i,j]];
            }
        };
        assert_eq!(
            Matrix::<f64>::geam(1.0, &a, -1.0, &b).as_slice(),
            r.as_slice());

        assert_eq!(
            Matrix::<f64>::geam(1.0, &a, -1.0, &a_t.t()).as_slice(),
            zero_mat.as_slice());

        assert_eq!(
            Matrix::<f64>::geam(1.0, &a_t.t(), -1.0, &a).as_slice(),
            zero_mat_t.t().as_slice());


        // Mutable geam

        let mut c = Matrix::<f64>::geam(1.0, &a, 1.0, &b);
        Matrix::<f64>::geam_mut(-1.0, &a, 1.0, &(c.clone()), &mut c);
        assert_eq!( c.as_slice(), b.as_slice());

        for (alpha, beta) in [ (1.0,1.0), (1.0,-1.0), (-1.0,-1.0), (-1.0,1.0),
                               (1.0,0.0), (0.0,-1.0), (0.5, 1.0), (0.5, 1.0),
                               (0.5,-0.5) ] {
            let mut c = a.clone();
            Matrix::<f64>::geam_mut(alpha, &a, beta, &b, &mut c);
            assert_eq!( c.as_slice(),
                Matrix::<f64>::geam(alpha, &a, beta, &b).as_slice());
        };
    }


}
