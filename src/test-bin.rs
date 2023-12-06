include!("common.rs");

extern crate rayon;
use rayon::prelude::*;

const DO_BLAS : bool = false;
const NMAX : usize =2;

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
    let k = 3030*NMAX;

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

    let duration = time.elapsed();
    println!("Time elapsed in preparation: {:?}", duration);


    // BLAS DGEMM
    if DO_BLAS {
        let time = std::time::Instant::now();
        $blas_gemm(b'N', b'N', m, n, k, 2.0, &a, m, &b, k, 0.0, &mut c_ref, m);

        let duration = time.elapsed();
        println!("Time elapsed in BLAS: {:?}", duration);
    }


    // Tiling
    let time = std::time::Instant::now();

    let a_mat = TiledMatrix::<$s>::from(&a, m, k, m);
    let b_mat = TiledMatrix::<$s>::from(&b, k, n, k);

    let duration = time.elapsed();
    println!("Time elapsed in tiling: {:?}", duration);


    // GEMM
    let time = std::time::Instant::now();

    let _ = TiledMatrix::<$s>::gemm(2.0, &a_mat, &b_mat);
    let duration = time.elapsed();

    println!("Time elapsed in CPU gemm: {:?}", duration);
    drop(a_mat);
    drop(b_mat);

    // Tiling
    let time = std::time::Instant::now();

    let a_mat = TiledMatrixGPU::<$s>::from(&a, m, k, m);
    let b_mat = TiledMatrixGPU::<$s>::from(&b, k, n, k);

    let duration = time.elapsed();
    println!("Time elapsed in tiling: {:?}", duration);

    // GEMM
    let time = std::time::Instant::now();

    let c = TiledMatrixGPU::<$s>::gemm(2.0, &a_mat, &b_mat);
    let duration = time.elapsed();

    println!("Time elapsed in GPU gemm: {:?}", duration);


    // Untiling
    let mut c_vec = vec![ 0. ; m*n ];
    let time = std::time::Instant::now();

    c.copy_in_vec(&mut c_vec, m);

    let duration = time.elapsed();
    println!("Time elapsed in untiling: {:?}", duration);

    if DO_BLAS {
        assert_eq!(c_vec, c_ref);
    }

}
}
}

impl_main!(f64, time_dgemm, blas_utils::dgemm);
impl_main!(f32, time_sgemm, blas_utils::sgemm);

