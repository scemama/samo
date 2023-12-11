include!("common.rs");

extern crate rayon;
use rayon::prelude::*;
use crate::cuda::Device;

const DO_BLAS : bool = false;
const NMAX : usize =1;

fn main() {
    time_sgemm();
    time_dgemm();
}



macro_rules! impl_main {
  ($s:ty, $main_path:ident, $blas_gemm:path) => {
pub fn $main_path() {

    // Preparation
    let m = 10100*NMAX;
    let n = 20200*NMAX;
    let k = 30300*NMAX;

    let time = std::time::Instant::now();

    let mut a = vec![ 0. ; m*k ];
    a.par_chunks_mut(m).enumerate().for_each(|(j,x)| {
        for i in 0..m {
            x[i] = (i as $s) + (j as $s)*10.0;
        }
    });

    let mut b = vec![ 0. ; k*n ];
    b.par_chunks_mut(k).enumerate().for_each(|(j,x)| {
        for i in 0..k {
            x[i] = -(i as $s) + (j as $s)*7.0;
        }
    });

    let mut c_ref = vec![ 0. ; m*n ];

    let a_mat = Matrix::<$s>::from(a.as_ptr(), m, k, m);
    let b_mat = Matrix::<$s>::from(b.as_ptr(), k, n, k);

    let duration = time.elapsed();
    println!("Time elapsed in preparation: {:?}", duration);


    // BLAS DGEMM
    if DO_BLAS {
        let time = std::time::Instant::now();
        $blas_gemm(b'N', b'N', m, n, k, 2.0, &a, m, &b, k, 0.0, &mut c_ref, m);

        let duration = time.elapsed();
        println!("Time elapsed in BLAS: {:?}", duration);
    }

    // GEMM
    let time = std::time::Instant::now();

    let _ = Matrix::<$s>::gemm(2.0, &a_mat, &b_mat);
    let duration = time.elapsed();

    println!("Time elapsed in CPU gemm: {:?}", duration);

    // Preparation
    let time = std::time::Instant::now();

    let mut a_mat_gpu = Matrix::<$s>::new(Device::CPU, m, k);
    for j in 0..k {
      for i in 0..m {
        a_mat_gpu[[i,j]] = a_mat[[i,j]]
      }
    }
    a_mat_gpu.prefetch(Device::GPU(0));

    let mut b_mat_gpu = Matrix::<$s>::new(Device::CPU, k, n);
    for j in 0..n {
      for i in 0..k {
        b_mat_gpu[[i,j]] = b_mat[[i,j]]
      }
    }
    b_mat_gpu.prefetch(Device::GPU(0));

    Device::GPU(0).synchronize();

    let duration = time.elapsed();
    println!("Time elapsed in preparation: {:?}", duration);

    // GEMM
    let time = std::time::Instant::now();

    let _ = Matrix::<$s>::gemm(2.0, &a_mat, &b_mat);
    let duration = time.elapsed();

    println!("Time elapsed in GPU gemm: {:?}", duration);


}
}
}

impl_main!(f64, time_dgemm, blas_utils::dgemm);
impl_main!(f32, time_sgemm, blas_utils::sgemm);

