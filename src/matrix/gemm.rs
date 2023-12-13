use super::*;

use crate::blas_utils;
use crate::cuda;
use cuda::{Device, DevPtr};

use crate::cublas;

macro_rules! impl_matrix {
($s:ty, $gemm:ident, $gemm_cpu:path, $gemm_gpu:path) => {

impl Matrix<$s>
{

   fn recursive_gemm_nn(handle: &cublas::Context, m:usize, n:usize, k:usize, alpha: $s,
              a_ptr: &DevPtr<$s>, lda:usize,
              b_ptr: &DevPtr<$s>, ldb:usize, beta: $s,
              c_ptr: &mut DevPtr<$s>, ldc:usize, block_size: usize) {

      if (n > block_size) {

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

      } else if (m > block_size) {

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

       let chunk_size = block_size;
       let n_chunks = k/chunk_size+1 ;

       for i in 0..n_chunks {

            let offset = i * chunk_size;
            let current_k = if i < n_chunks - 1 { chunk_size } else { k - offset };

            let a_offset_ptr = a_ptr.offset((lda*offset) as isize);
            let b_offset_ptr = b_ptr.offset(offset as isize);

            a_offset_ptr.prefetch_only(lda * (current_k-1) + m);
            b_offset_ptr.prefetch_only(ldb * (n-1) + current_k);

            $gemm_gpu(handle, b'N', b'N', m, n, current_k, alpha,
                      &a_offset_ptr, lda, &b_offset_ptr, ldb,
                      if i == 0 { beta } else { 1.0 }, c_ptr, ldc).unwrap();
            }

       }
   }

   fn recursive_gemm_tn(handle: &cublas::Context, m:usize, n:usize, k:usize, alpha: $s,
              a_ptr: &DevPtr<$s>, lda:usize,
              b_ptr: &DevPtr<$s>, ldb:usize, beta: $s,
              c_ptr: &mut DevPtr<$s>, ldc:usize, block_size: usize) {

      if (n > block_size) {

        let n1 = n/2;
        let n2 = n - n1;
        let b_ptr2 = b_ptr.offset((ldb*n1) as isize);
        let mut c_ptr2 = c_ptr.offset((ldc*n1) as isize);

        Self::recursive_gemm_tn(handle, m, n1, k,
                  alpha, &a_ptr, lda, &b_ptr, ldb, beta,
                  c_ptr, ldc, block_size);

        Self::recursive_gemm_tn(handle, m, n2, k,
                  alpha, &a_ptr, lda, &b_ptr2, ldb, beta,
                  &mut c_ptr2, ldc, block_size);

      } else if (m > block_size) {

        let m1 = m/2;
        let m2 = m - m1;
        let a_ptr2 = a_ptr.offset((lda*m1) as isize);
        let mut c_ptr2 = c_ptr.offset(m1 as isize);

        Self::recursive_gemm_tn(handle, m1, n, k,
                  alpha, &a_ptr, lda, &b_ptr, ldb, beta,
                  c_ptr, ldc, block_size);

        Self::recursive_gemm_tn(handle, m2, n, k,
                  alpha, &a_ptr2, lda, &b_ptr, ldb, beta,
                  &mut c_ptr2, ldc, block_size);

      } else {


       c_ptr.prefetch_only(ldc * (n - 1) + m);

       let chunk_size = block_size;
       let n_chunks = k/chunk_size+1 ;

       for i in 0..n_chunks {

            let offset = i * chunk_size;
            let current_k = if i < n_chunks - 1 { chunk_size } else { k - offset };

            let a_offset_ptr = a_ptr.offset(offset as isize);
            let b_offset_ptr = b_ptr.offset(offset as isize);

            a_offset_ptr.prefetch_only(lda * (m-1) + current_k);
            b_offset_ptr.prefetch_only(ldb * (n-1) + current_k);

            $gemm_gpu(handle, b'T', b'N', m, n, current_k, alpha,
                      &a_offset_ptr, lda, &b_offset_ptr, ldb,
                      if i == 0 { beta } else { 1.0 }, c_ptr, ldc).unwrap();
            }

       }
   }

   fn recursive_gemm_nt(handle: &cublas::Context, m:usize, n:usize, k:usize, alpha: $s,
              a_ptr: &DevPtr<$s>, lda:usize,
              b_ptr: &DevPtr<$s>, ldb:usize, beta: $s,
              c_ptr: &mut DevPtr<$s>, ldc:usize, block_size: usize) {

      if (n > block_size) {

        let n1 = n/2;
        let n2 = n - n1;
        let b_ptr2 = b_ptr.offset(n1 as isize);
        let mut c_ptr2 = c_ptr.offset((ldc*n1) as isize);

        Self::recursive_gemm_nt(handle, m, n1, k,
                  alpha, &a_ptr, lda, &b_ptr, ldb, beta,
                  c_ptr, ldc, block_size);

        Self::recursive_gemm_nt(handle, m, n2, k,
                  alpha, &a_ptr, lda, &b_ptr2, ldb, beta,
                  &mut c_ptr2, ldc, block_size);

      } else if (m > block_size) {

        let m1 = m/2;
        let m2 = m - m1;
        let a_ptr2 = a_ptr.offset(m1 as isize);
        let mut c_ptr2 = c_ptr.offset(m1 as isize);

        Self::recursive_gemm_nt(handle, m1, n, k,
                  alpha, &a_ptr, lda, &b_ptr, ldb, beta,
                  c_ptr, ldc, block_size);

        Self::recursive_gemm_nt(handle, m2, n, k,
                  alpha, &a_ptr2, lda, &b_ptr, ldb, beta,
                  &mut c_ptr2, ldc, block_size);

      } else {


       c_ptr.prefetch_only(ldc * (n - 1) + m);

       let chunk_size = block_size;
       let n_chunks = k/chunk_size+1 ;

       for i in 0..n_chunks {

            let offset = i * chunk_size;
            let current_k = if i < n_chunks - 1 { chunk_size } else { k - offset };

            let a_offset_ptr = a_ptr.offset((lda*offset) as isize);
            let b_offset_ptr = b_ptr.offset((ldb*offset) as isize);

            a_offset_ptr.prefetch_only(lda * (current_k-1) + m );
            b_offset_ptr.prefetch_only(ldb * (current_k-1) + n );

            $gemm_gpu(handle, b'N', b'T', m, n, current_k, alpha,
                      &a_offset_ptr, lda, &b_offset_ptr, ldb,
                      if i == 0 { beta } else { 1.0 }, c_ptr, ldc).unwrap();
            }

       }
   }

   fn recursive_gemm_tt(handle: &cublas::Context, m:usize, n:usize, k:usize, alpha: $s,
              a_ptr: &DevPtr<$s>, lda:usize,
              b_ptr: &DevPtr<$s>, ldb:usize, beta: $s,
              c_ptr: &mut DevPtr<$s>, ldc:usize, block_size: usize) {

      if (n > block_size) {

        let n1 = n/2;
        let n2 = n - n1;
        let b_ptr2 = b_ptr.offset(n1 as isize);
        let mut c_ptr2 = c_ptr.offset((ldc*n1) as isize);

        Self::recursive_gemm_tt(handle, m, n1, k,
                  alpha, &a_ptr, lda, &b_ptr, ldb, beta,
                  c_ptr, ldc, block_size);

        Self::recursive_gemm_tt(handle, m, n2, k,
                  alpha, &a_ptr, lda, &b_ptr2, ldb, beta,
                  &mut c_ptr2, ldc, block_size);

      } else if (m > block_size) {

        let m1 = m/2;
        let m2 = m - m1;
        let a_ptr2 = a_ptr.offset((lda*m1) as isize);
        let mut c_ptr2 = c_ptr.offset(m1 as isize);

        Self::recursive_gemm_tt(handle, m1, n, k,
                  alpha, &a_ptr, lda, &b_ptr, ldb, beta,
                  c_ptr, ldc, block_size);

        Self::recursive_gemm_tt(handle, m2, n, k,
                  alpha, &a_ptr2, lda, &b_ptr, ldb, beta,
                  &mut c_ptr2, ldc, block_size);

      } else {


       c_ptr.prefetch_only(ldc * (n - 1) + m);

       let chunk_size = block_size;
       let n_chunks = k/chunk_size+1 ;

       for i in 0..n_chunks {

            let offset = i * chunk_size;
            let current_k = if i < n_chunks - 1 { chunk_size } else { k - offset };

            let a_offset_ptr = a_ptr.offset(offset as isize);
            let b_offset_ptr = b_ptr.offset((ldb*offset) as isize);

            a_offset_ptr.prefetch_only(lda * (m-1) + current_k );
            b_offset_ptr.prefetch_only(ldb * (current_k-1) + n );

            $gemm_gpu(handle, b'T', b'T', m, n, current_k, alpha,
                      &a_offset_ptr, lda, &b_offset_ptr, ldb,
                      if i == 0 { beta } else { 1.0 }, c_ptr, ldc).unwrap();
            }

       }
   }


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
            let block_size: usize = num::integer::sqrt(mem.total)/32;
            let lda = a.lda;
            let ldb = b.lda;
            let ldc = c.lda;
            match (a.transposed, b.transposed) {
                (false, false) => Self::recursive_gemm_nn(&handle, c.nrows, c.ncols, b.nrows(),
                                      alpha, a_ptr, a.lda, b_ptr, b.lda, beta,
                                      c_ptr, ldc, block_size),
                (true, false) => Self::recursive_gemm_tn(&handle, c.nrows, c.ncols, b.nrows(),
                                      alpha, a_ptr, a.lda, b_ptr, b.lda, beta,
                                      c_ptr, ldc, block_size),
                (false, true) => Self::recursive_gemm_nt(&handle, c.nrows, c.ncols, b.nrows(),
                                      alpha, a_ptr, a.lda, b_ptr, b.lda, beta,
                                      c_ptr, ldc, block_size),
                (true, true ) =>  Self::recursive_gemm_tt(&handle, c.nrows, c.ncols, b.nrows(),
                                      alpha, a_ptr, a.lda, b_ptr, b.lda, beta,
                                      c_ptr, ldc, block_size),
            };
          },

         _ => {  // Run on CPU
            $gemm_cpu(transa, transb, c.nrows, c.ncols, b.nrows(),
                    alpha, a.as_slice(), a.lda, b.as_slice(), b.lda, beta,
                    c.as_slice_mut(), ldc)
          },

       };
   }
} // end impl Matrix

//--------------------------------------------------------------

#[cfg(test)]
mod $gemm {
    use super::*;

    #[test]
    fn test() {
        let mut a_vec = vec![1. , 1.1, 1.2, 1.3,
                         2. , 2.1, 2.2, 2.3,
                         3. , 3.1, 3.2, 3.3];

        let mut b_vec = vec![ 1.0, 2.0, 3.0, 4.0,
                          1.1, 2.1, 3.1, 4.1];

        let mut c_vec = vec![12.0 , 22.0 , 32.0,
                         12.46, 22.86, 33.26];

        let a = Matrix::<$s>::from( a_vec.as_mut_ptr(), 4, 3, 4).t();
        let b = Matrix::<$s>::from( b_vec.as_mut_ptr(), 4, 2, 4 );
        let c_ref = Matrix::<$s>::from( c_vec.as_mut_ptr(), 3, 2, 3);

        let c = Matrix::<$s>::gemm(1.0, &a, &b);

        let difference = Matrix::<$s>::geam(1.0, &c, -1.0, &c_ref);
        for j in 0..2 {
            for i in 0..3 {
                assert!(num::abs(difference[[i,j]] / c[[i,j]]) < <$s>::EPSILON);
            }
        }

        let a = a.t();
        let b = b.t();
        let c_t = Matrix::<$s>::gemm(1.0, &b, &a);
        let c_t2 = c.t();
        for j in 0..3 {
            for i in 0..2 {
                assert_eq!(c_t[[i,j]], c_t2[[i,j]]);
            }
        }



        let a_gpu = Matrix::<$s>::new(Device::GPU(0), 4, 3).t();
        let b_gpu = Matrix::<$s>::new(Device::GPU(0), 4, 2);
        let c_gpu = Matrix::<$s>::gemm(1.0, &a_gpu, &b_gpu);

        let difference = Matrix::<$s>::geam(1.0, &c, -1.0, &c_ref);
        for j in 0..2 {
            for i in 0..3 {
                assert!(num::abs(difference[[i,j]] / c[[i,j]]) < <$s>::EPSILON);
            }
        }

        let a_gpu = a_gpu.t();
        let b_gpu = b_gpu.t();
        let c_t = Matrix::<$s>::gemm(1.0, &b_gpu, &a_gpu);
        let c_t2 = c_gpu.t();
        for j in 0..3 {
            for i in 0..2 {
                assert_eq!(c_t[[i,j]], c_t2[[i,j]]);
            }
        }

    }
}

}} // end macro

impl_matrix!(f32, f32_gemm, blas_utils::sgemm, cublas::sgemm);
impl_matrix!(f64, f64_gemm, blas_utils::dgemm, cublas::dgemm);

//--------------------------------------------------------------

