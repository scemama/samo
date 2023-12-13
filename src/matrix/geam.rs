use super::*;

use core::iter::zip;
use crate::cuda;
use cuda::{Device, DevPtr};
use crate::cublas;

macro_rules! impl_matrix {
($s:ty, $geam:ident, $geam_gpu:path) => {

impl Matrix<$s>
{

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

//--------------------------------------------------------------

#[cfg(test)]
mod $geam {
    use super::*;

    #[test]
    fn test() {
        let n = 8;
        let m = 4;
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
            Matrix::<$s>::geam(0.0, &a, 0.0, &b).as_slice(),
            zero_mat.as_slice());

        assert_eq!(
            Matrix::<$s>::geam(1.0, &a, 0.0, &b).as_slice(),
            a.as_slice());

        assert_eq!(
            Matrix::<$s>::geam(0.0, &a, 1.0, &b).as_slice(),
            b.as_slice());

        let mut r = Matrix::<$s>::new(Device::CPU, m, n);
        for i in 0..m {
            for j in 0..n {
                r[[i,j]] = 2.0*a[[i,j]];
            }
        };
        assert_eq!(
            Matrix::<$s>::geam(2.0, &a, 0.0, &b).as_slice(),
            r.as_slice());

        assert_eq!(
            Matrix::<$s>::geam(0.5, &a, 0.5, &a).as_slice(),
            a.as_slice());

        assert_eq!(
            Matrix::<$s>::geam(0.5, &a, -0.5, &a).as_slice(),
            zero_mat.as_slice());
        assert_eq!(
            Matrix::<$s>::geam(1.0, &a, -1.0, &a).as_slice(),
            zero_mat.as_slice());

        let mut r = Matrix::<$s>::new(Device::CPU, m, n);
        for i in 0..m {
            for j in 0..n {
                r[[i,j]] = 2.0*b[[i,j]];
            }
        };
        println!("{:?}", a);
        println!("{:?}", b);
        assert_eq!(
            Matrix::<$s>::geam(0.0, &a, 2.0, &b).as_slice(),
            r.as_slice() );

        let mut r = Matrix::<$s>::new(Device::CPU, m, n);
        for i in 0..m {
            for j in 0..n {
                r[[i,j]] = a[[i,j]] + b[[i,j]];
            }
        };
        assert_eq!(
            Matrix::<$s>::geam(1.0, &a, 1.0, &b).as_slice(),
            r.as_slice());

        let mut r = Matrix::<$s>::new(Device::CPU, m, n);
        for i in 0..m {
            for j in 0..n {
                r[[i,j]] = b[[i,j]] - a[[i,j]];
            }
        };
        assert_eq!(
            Matrix::<$s>::geam(-1.0, &a, 1.0, &b).as_slice(),
            r.as_slice());

        let mut r = Matrix::<$s>::new(Device::CPU, m, n);
        for i in 0..m {
            for j in 0..n {
                r[[i,j]] = a[[i,j]] - b[[i,j]];
            }
        };
        assert_eq!(
            Matrix::<$s>::geam(1.0, &a, -1.0, &b).as_slice(),
            r.as_slice());

        assert_eq!(
            Matrix::<$s>::geam(1.0, &a, -1.0, &a_t.t()).as_slice(),
            zero_mat.as_slice());

        assert_eq!(
            Matrix::<$s>::geam(1.0, &a_t.t(), -1.0, &a).as_slice(),
            zero_mat_t.t().as_slice());


        // Mutable geam

        let mut c = Matrix::<$s>::geam(1.0, &a, 1.0, &b);
        Matrix::<$s>::geam_mut(-1.0, &a, 1.0, &(c.clone()), &mut c);
        assert_eq!( c.as_slice(), b.as_slice());

        for (alpha, beta) in [ (1.0,1.0), (1.0,-1.0), (-1.0,-1.0), (-1.0,1.0),
                               (1.0,0.0), (0.0,-1.0), (0.5, 1.0), (0.5, 1.0),
                               (0.5,-0.5) ] {
            let mut c = a.clone();
            Matrix::<$s>::geam_mut(alpha, &a, beta, &b, &mut c);
            assert_eq!( c.as_slice(),
                Matrix::<$s>::geam(alpha, &a, beta, &b).as_slice());
        };
    }

}

}} // end macro

impl_matrix!(f32, f32_geam, cublas::sgeam);
impl_matrix!(f64, f64_geam, cublas::dgeam);

