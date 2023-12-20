use super::*;

use core::iter::zip;
use crate::cuda;
use cuda::{Device, DevPtr};
use crate::cublas;

macro_rules! impl_matrix {
($s:ty, $geam:ident, $geam_gpu:path) => {

impl Matrix<$s>
{

   fn recursive_geam_nn(handle: &cublas::Context, m:usize, n:usize,
           alpha: $s, a_ptr: &DevPtr<$s>, lda:usize,
           beta:  $s, b_ptr: &DevPtr<$s>, ldb:usize,
           c_ptr: &mut DevPtr<$s>, ldc:usize, block_size: usize) {
      if lda*n > block_size || ldb*n > block_size || ldc*n > block_size {

        let n1 = n/2;
        let n2 = n - n1;
        let a_ptr2 = a_ptr.offset((lda*n1) as isize);
        let b_ptr2 = b_ptr.offset((ldb*n1) as isize);
        let mut c_ptr2 = c_ptr.offset((ldc*n1) as isize);

        Self::recursive_geam_nn(handle, m, n1,
                  alpha, &a_ptr, lda, beta, &b_ptr, ldb,
                  c_ptr, ldc, block_size);

        Self::recursive_geam_nn(handle, m, n2,
                  alpha, &a_ptr2, lda, beta, &b_ptr2, ldb,
                  &mut c_ptr2, ldc, block_size);

      } else if m*n > block_size {

        let m1 = m/2;
        let m2 = m - m1;
        let a_ptr2 = a_ptr.offset(m1 as isize);
        let b_ptr2 = b_ptr.offset(m1 as isize);
        let mut c_ptr2 = c_ptr.offset(m1 as isize);

        Self::recursive_geam_nn(handle, m1, n,
                  alpha, &a_ptr, lda, beta, &b_ptr, ldb,
                  c_ptr, ldc, block_size);

        Self::recursive_geam_nn(handle, m2, n,
                  alpha, &a_ptr2, lda, beta, &b_ptr2, ldb,
                  &mut c_ptr2, ldc, block_size);

      } else {

        a_ptr.prefetch_only(lda * (n - 1) + m);
        b_ptr.prefetch_only(ldb * (n - 1) + m);
        c_ptr.prefetch_only(ldc * (n - 1) + m);

        $geam_gpu(handle, b'N', b'N', m, n, alpha,
                  &a_ptr, lda, beta, &b_ptr, ldb,
                  c_ptr, ldc).unwrap();
       }
   }

   fn recursive_geam_tn(handle: &cublas::Context, m:usize, n:usize,
           alpha: $s, a_ptr: &DevPtr<$s>, lda:usize,
           beta:  $s, b_ptr: &DevPtr<$s>, ldb:usize,
           c_ptr: &mut DevPtr<$s>, ldc:usize, block_size: usize) {
      if ldb*n > block_size || ldc*n > block_size {

        let n1 = n/2;
        let n2 = n - n1;
        let a_ptr2 = a_ptr.offset(n1 as isize);
        let b_ptr2 = b_ptr.offset((ldb*n1) as isize);
        let mut c_ptr2 = c_ptr.offset((ldc*n1) as isize);

        Self::recursive_geam_tn(handle, m, n1,
                  alpha, &a_ptr, lda, beta, &b_ptr, ldb,
                  c_ptr, ldc, block_size);

        Self::recursive_geam_tn(handle, m, n2,
                  alpha, &a_ptr2, lda, beta, &b_ptr2, ldb,
                  &mut c_ptr2, ldc, block_size);

      } else if lda*m > block_size {

        let m1 = m/2;
        let m2 = m - m1;
        let a_ptr2 = a_ptr.offset((lda*m1) as isize);
        let b_ptr2 = b_ptr.offset(m1 as isize);
        let mut c_ptr2 = c_ptr.offset(m1 as isize);

        Self::recursive_geam_tn(handle, m1, n,
                  alpha, &a_ptr, lda, beta, &b_ptr, ldb,
                  c_ptr, ldc, block_size);

        Self::recursive_geam_tn(handle, m2, n,
                  alpha, &a_ptr2, lda, beta, &b_ptr2, ldb,
                  &mut c_ptr2, ldc, block_size);

      } else {

        a_ptr.prefetch_only(lda * (m - 1) + n);
        b_ptr.prefetch_only(ldb * (n - 1) + m);
        c_ptr.prefetch_only(ldc * (n - 1) + m);

        $geam_gpu(handle, b'T', b'N', m, n, alpha,
                  &a_ptr, lda, beta, &b_ptr, ldb,
                  c_ptr, ldc).unwrap();
       }
   }

   fn recursive_geam_nt(handle: &cublas::Context, m:usize, n:usize,
           alpha: $s, a_ptr: &DevPtr<$s>, lda:usize,
           beta:  $s, b_ptr: &DevPtr<$s>, ldb:usize,
           c_ptr: &mut DevPtr<$s>, ldc:usize, block_size: usize) {
      if lda*n > block_size || ldc*n > block_size {

        let n1 = n/2;
        let n2 = n - n1;
        let a_ptr2 = a_ptr.offset((lda*n1) as isize);
        let b_ptr2 = b_ptr.offset(n1 as isize);
        let mut c_ptr2 = c_ptr.offset((ldc*n1) as isize);

        Self::recursive_geam_nt(handle, m, n1,
                  alpha, &a_ptr, lda, beta, &b_ptr, ldb,
                  c_ptr, ldc, block_size);

        Self::recursive_geam_nt(handle, m, n2,
                  alpha, &a_ptr2, lda, beta, &b_ptr2, ldb,
                  &mut c_ptr2, ldc, block_size);

      } else if ldb*m > block_size {

        let m1 = m/2;
        let m2 = m - m1;
        let a_ptr2 = a_ptr.offset(m1 as isize);
        let b_ptr2 = b_ptr.offset((ldb*m1) as isize);
        let mut c_ptr2 = c_ptr.offset(m1 as isize);

        Self::recursive_geam_nt(handle, m1, n,
                  alpha, &a_ptr, lda, beta, &b_ptr, ldb,
                  c_ptr, ldc, block_size);

        Self::recursive_geam_nt(handle, m2, n,
                  alpha, &a_ptr2, lda, beta, &b_ptr2, ldb,
                  &mut c_ptr2, ldc, block_size);

      } else {

        a_ptr.prefetch_only(lda * (n - 1) + m);
        b_ptr.prefetch_only(ldb * (m - 1) + n);
        c_ptr.prefetch_only(ldc * (n - 1) + m);

        $geam_gpu(handle, b'N', b'T', m, n, alpha,
                  &a_ptr, lda, beta, &b_ptr, ldb,
                  c_ptr, ldc).unwrap();
       }
   }

   fn recursive_geam_tt(handle: &cublas::Context, m:usize, n:usize,
           alpha: $s, a_ptr: &DevPtr<$s>, lda:usize,
           beta:  $s, b_ptr: &DevPtr<$s>, ldb:usize,
           c_ptr: &mut DevPtr<$s>, ldc:usize, block_size: usize) {
      if ldc*n > block_size {

        let n1 = n/2;
        let n2 = n - n1;
        let a_ptr2 = a_ptr.offset(n1 as isize);
        let b_ptr2 = b_ptr.offset(n1 as isize);
        let mut c_ptr2 = c_ptr.offset((ldc*n1) as isize);

        Self::recursive_geam_tt(handle, m, n1,
                  alpha, &a_ptr, lda, beta, &b_ptr, ldb,
                  c_ptr, ldc, block_size);

        Self::recursive_geam_tt(handle, m, n2,
                  alpha, &a_ptr2, lda, beta, &b_ptr2, ldb,
                  &mut c_ptr2, ldc, block_size);

      } else if lda*m > block_size || ldb*m > block_size {

        let m1 = m/2;
        let m2 = m - m1;
        let a_ptr2 = a_ptr.offset((lda*m1) as isize);
        let b_ptr2 = b_ptr.offset((ldb*m1) as isize);
        let mut c_ptr2 = c_ptr.offset(m1 as isize);

        Self::recursive_geam_tt(handle, m1, n,
                  alpha, &a_ptr, lda, beta, &b_ptr, ldb,
                  c_ptr, ldc, block_size);

        Self::recursive_geam_tt(handle, m2, n,
                  alpha, &a_ptr2, lda, beta, &b_ptr2, ldb,
                  &mut c_ptr2, ldc, block_size);

      } else {

        a_ptr.prefetch_only(lda * (m - 1) + n);
        b_ptr.prefetch_only(ldb * (m - 1) + n);
        c_ptr.prefetch_only(ldc * (n - 1) + m);

        $geam_gpu(handle, b'T', b'T', m, n, alpha,
                  &a_ptr, lda, beta, &b_ptr, ldb,
                  c_ptr, ldc).unwrap();
       }
   }

   fn geam_cpu(alpha: $s, a: &Self, beta: $s, b: &Self, c: &mut Self) {
       let nrows = c.nrows;
       let ncols = c.ncols;
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

           (1,0,false,_) =>  {
               for (x, &v) in zip(c.as_slice_mut(), a.as_slice()) {
                   *x = v;
               }
           },

           (-1,0,false,_) =>  {
               for (x, &v) in zip(c.as_slice_mut(), a.as_slice()) {
                   *x = -v;
               }
           },

           (_,0,false,_) =>  {
               for (x, &v) in zip(c.as_slice_mut(), a.as_slice()) {
                   *x = alpha*v;
               }
           },

           (0,1,_,false) =>  {
               for (x, &v) in zip(c.as_slice_mut(), b.as_slice()) {
                   *x = v;
               }
           },

           (0,-1,_,false) =>  {
               for (x, &v) in zip(c.as_slice_mut(), b.as_slice()) {
                   *x = -v;
               }
           },

           (0,_,_,false) =>  {
               for (x, &v) in zip(c.as_slice_mut(), b.as_slice()) {
                   *x = beta*v;
               }
           },

           (1, 1, false, false) => {
               for (x, (&v, &w)) in zip(c.as_slice_mut(), zip(a.as_slice(), b.as_slice())) {
                   *x = v + w;
               }},

           (1,-1, false, false) => {
               for (x, (&v, &w)) in zip(c.as_slice_mut(), zip(a.as_slice(), b.as_slice())) {
                   *x = v - w;
               }},

           (_, _, false, false) => {
               for (x, (&v, &w)) in zip(c.as_slice_mut(), zip(a.as_slice(), b.as_slice())) {
                   *x = alpha * v + beta * w;
               }},

           (_, _, true, false) => {
               let a_ = a.as_slice();
               let b_ = b.as_slice();
               let ldc = c.lda;
               let c_ = c.as_slice_mut();
               for i in 0..ncols   {
                   for j in 0..nrows   {
                       let x = a_[i+j*a.lda];
                       let y = b_[j+i*b.lda];
                       c_[j+i*ldc] = alpha * x + beta * y;
                   }
               }
              },

           (_, _, false, true) => {
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

           (_, _, true, true) => {
               let a_ = a.as_slice();
               let b_ = b.as_slice();
               let ldc = c.lda;
               let c_ = c.as_slice_mut();
               for j in 0..ncols   {
                   for i in 0..nrows   {
                   let x = a_[j+i*a.lda];
                   let y = b_[j+i*b.lda];
                   c_[i+j*ldc] = alpha * x + beta * y ;
                   }
               } },
       };
       c.transposed = false;
   }

   pub fn geam(handle: Option<&cublas::Context>,
               alpha: $s, a: &Self, beta: $s, b: &Self) -> Self {
       let device =
           match (a.device(), b.device()) {
                (Device::GPU(d), Device::GPU(_)) => Device::GPU(d),
                _ => Device::CPU,
            };
       let mut c = Self::new(device, a.nrows(), b.ncols());
       Self::geam_mut(handle, alpha, a, beta, b, &mut c);
       c
   }

   pub fn geam_mut(handle: Option<&cublas::Context>,
                   alpha: $s, a: &Self, beta: $s, b: &Self, c: &mut Self) {
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

       match (&a.data, &b.data, &mut c.data) {
         (Data::<$s>::GPU(a_ptr), Data::<$s>::GPU(b_ptr), Data::<$s>::GPU(c_ptr)) => {
                // Run on GPU
            let new_context = std::cell::OnceCell::new();
            let new_handle =
                match (handle) {
                    None => {
                        new_context.set(cublas::Context::new().unwrap()).unwrap();
                        &(new_context.get().unwrap())
                    },
                    Some(r) => r,
                };
            let mem = cuda::get_mem_info().unwrap();
            let block_size = mem.total / (8*4);
            let lda = a.lda;
            let ldb = b.lda;
            let ldc = c.lda;
            a_ptr.mem_advise(cuda::MemAdvise::SetReadMostly);
            b_ptr.mem_advise(cuda::MemAdvise::SetReadMostly);
            match (a.transposed, b.transposed) {
                (false, false) => Self::recursive_geam_nn(new_handle, c.nrows, c.ncols,
                                      alpha, a_ptr, a.lda, beta, b_ptr, b.lda,
                                      c_ptr, ldc, block_size),
                (true, false) => Self::recursive_geam_tn(new_handle, c.nrows, c.ncols,
                                      alpha, a_ptr, a.lda, beta, b_ptr, b.lda,
                                      c_ptr, ldc, block_size),
                (false, true) => Self::recursive_geam_nt(new_handle, c.nrows, c.ncols,
                                      alpha, a_ptr, a.lda, beta, b_ptr, b.lda,
                                      c_ptr, ldc, block_size),
                (true, true ) => Self::recursive_geam_tt(new_handle, c.nrows, c.ncols,
                                      alpha, a_ptr, a.lda, beta, b_ptr, b.lda,
                                      c_ptr, ldc, block_size),
            };
          },

         _ => {  // Run on CPU
            Self::geam_cpu(alpha, a, beta, b, c)
          },

       };

   }

} // end impl Matrix

//--------------------------------------------------------------

#[cfg(test)]
mod $geam {
    use super::*;

    #[test]
    fn test() {
        let n = 8;
        let m = 4;
        let ctx = cublas::Context::new();
        for device in [ Device::CPU, Device::GPU(0) ] {
            let handle =
              match device {
                Device::CPU => None,
                _ => Some(&ctx),
              };
            let mut a   = Matrix::<$s>::new(Device::CPU, m, n);
            let mut a_t = Matrix::<$s>::new(Device::CPU, n, m);
            let mut b   = Matrix::<$s>::new(Device::CPU, m, n);
            let mut b_t = Matrix::<$s>::new(Device::CPU, n, m);
            let zero_mat = Matrix::<$s>::new(Device::CPU, m, n);
            let zero_mat_t = Matrix::<$s>::new(Device::CPU, n, m);
            for i in 0..m {
                for j in 0..n {
                    a[[i,j]] = (i * 10 + j) as $s;
                    b[[i,j]] = (i * 1000 + j*10) as $s;
                    a_t[[j,i]] = (i * 10 + j) as $s;
                    b_t[[j,i]] = (i * 1000 + j*10) as $s;
                }
            }

            assert_eq!(
                Matrix::<$s>::geam(handle, 0.0, &a, 0.0, &b).as_slice(),
                zero_mat.as_slice());

            assert_eq!(
                Matrix::<$s>::geam(handle, 1.0, &a, 0.0, &b).as_slice(),
                a.as_slice());

            assert_eq!(
                Matrix::<$s>::geam(handle, 0.0, &a, 1.0, &b).as_slice(),
                b.as_slice());

            let mut r = Matrix::<$s>::new(Device::CPU, m, n);
            for i in 0..m {
                for j in 0..n {
                    r[[i,j]] = 2.0*a[[i,j]];
                }
            };
            assert_eq!(
                Matrix::<$s>::geam(handle, 2.0, &a, 0.0, &b).as_slice(),
                r.as_slice());

            assert_eq!(
                Matrix::<$s>::geam(handle, 0.5, &a, 0.5, &a).as_slice(),
                a.as_slice());

            assert_eq!(
                Matrix::<$s>::geam(handle, 0.5, &a, -0.5, &a).as_slice(),
                zero_mat.as_slice());
            assert_eq!(
                Matrix::<$s>::geam(handle, 1.0, &a, -1.0, &a).as_slice(),
                zero_mat.as_slice());

            let mut r = Matrix::<$s>::new(Device::CPU, m, n);
            for i in 0..m {
                for j in 0..n {
                    r[[i,j]] = 2.0*b[[i,j]];
                }
            };
            assert_eq!(
                Matrix::<$s>::geam(handle, 0.0, &a, 2.0, &b).as_slice(),
                r.as_slice() );

            let mut r = Matrix::<$s>::new(Device::CPU, m, n);
            for i in 0..m {
                for j in 0..n {
                    r[[i,j]] = a[[i,j]] + b[[i,j]];
                }
            };
            assert_eq!(
                Matrix::<$s>::geam(handle, 1.0, &a, 1.0, &b).as_slice(),
                r.as_slice());

            let mut r = Matrix::<$s>::new(Device::CPU, m, n);
            for i in 0..m {
                for j in 0..n {
                    r[[i,j]] = b[[i,j]] - a[[i,j]];
                }
            };
            assert_eq!(
                Matrix::<$s>::geam(handle, -1.0, &a, 1.0, &b).as_slice(),
                r.as_slice());

            let mut r = Matrix::<$s>::new(Device::CPU, m, n);
            for i in 0..m {
                for j in 0..n {
                    r[[i,j]] = a[[i,j]] - b[[i,j]];
                }
            };
            assert_eq!(
                Matrix::<$s>::geam(handle, 1.0, &a, -1.0, &b).as_slice(),
                r.as_slice());

            assert_eq!(
                Matrix::<$s>::geam(handle, 1.0, &a, -1.0, &a_t.t()).as_slice(),
                zero_mat.as_slice());

            assert_eq!(
                Matrix::<$s>::geam(handle, 1.0, &a_t.t(), -1.0, &a).as_slice(),
                zero_mat_t.t().as_slice());


            // Mutable geam

            let mut c = Matrix::<$s>::geam(handle, 1.0, &a, 1.0, &b);
            Matrix::<$s>::geam_mut(None, -1.0, &a, 1.0, &(c.clone()), &mut c);
            assert_eq!( c.as_slice(), b.as_slice());

            for (alpha, beta) in [ (1.0,1.0), (1.0,-1.0), (-1.0,-1.0), (-1.0,1.0),
                                (1.0,0.0), (0.0,-1.0), (0.5, 1.0), (0.5, 1.0),
                                (0.5,-0.5) ] {
                let mut c = a.clone();
                Matrix::<$s>::geam_mut(None, alpha, &a, beta, &b, &mut c);
                assert_eq!( c.as_slice(),
                    Matrix::<$s>::geam(handle, alpha, &a, beta, &b).as_slice());
            };
        }
    }

}

}} // end macro

impl_matrix!(f32, f32_geam, cublas::sgeam);
impl_matrix!(f64, f64_geam, cublas::dgeam);

