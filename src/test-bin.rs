include!("common.rs");

extern crate rayon;
use rayon::prelude::*;

const DO_BLAS : bool = false;
const NMAX : usize =2;

fn main() {
    time_sgemm();
    time_dgemm();
}


pub fn time_dgemm() {

    // Preparation
    let m = 10100*NMAX;
    let n = 20200*NMAX;
    let k = 3030*NMAX;

    let time = std::time::Instant::now();

    let mut a = vec![ 0. ; m*k ];
    a.par_chunks_mut(m).enumerate().for_each(|(j,x)| {
        for i in 0..m {
            x[i] = (i as f64) + (j as f64)*10.0;
        }
    });

    let mut b = vec![ 0. ; k*n ];
    b.par_chunks_mut(k).enumerate().for_each(|(j,x)| {
        for i in 0..k {
            x[i] = -(i as f64) + (j as f64)*7.0;
        }
    });

    let mut c_ref = vec![ 0. ; m*n ];

    let duration = time.elapsed();
    println!("Time elapsed in preparation: {:?}", duration);


    // BLAS DGEMM
    if DO_BLAS {
        let time = std::time::Instant::now();
        blas_utils::dgemm(b'N', b'N', m, n, k, 2.0, &a, m, &b, k, 0.0, &mut c_ref, m);

        let duration = time.elapsed();
        println!("Time elapsed in BLAS: {:?}", duration);
    }


    // Tiling
    let time = std::time::Instant::now();

    let a = TiledMatrix::<f64>::from(&a, m, k, m);
    let b = TiledMatrix::<f64>::from(&b, k, n, k);

    let duration = time.elapsed();
    println!("Time elapsed in tiling: {:?}", duration);


    // GEMM
    let time = std::time::Instant::now();

    let c = TiledMatrix::<f64>::gemm(2.0, &a, &b);
    let duration = time.elapsed();

    println!("Time elapsed in dgemm: {:?}", duration);

    // GEMM
    let time = std::time::Instant::now();

    let c = TiledMatrix::<f64>::gemm_gpu(2.0, &a, &b);
    let duration = time.elapsed();

    println!("Time elapsed in dgemm gpu: {:?}", duration);


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

pub fn time_sgemm() {


    // Preparation
    let m = 10100*NMAX;
    let n = 20200*NMAX;
    let k = 3030*NMAX;

    let time = std::time::Instant::now();

    let mut a = vec![ 0. ; m*k ];
    a.par_chunks_mut(m).enumerate().for_each(|(j,x)| {
        for i in 0..m {
            x[i] = (i as f32) + (j as f32)*10.0;
        }
    });

    let mut b = vec![ 0. ; k*n ];
    b.par_chunks_mut(k).enumerate().for_each(|(j,x)| {
        for i in 0..k {
            x[i] = -(i as f32) + (j as f32)*7.0;
        }
    });

    let mut c_ref = vec![ 0. ; m*n ];

    let duration = time.elapsed();
    println!("Time elapsed in preparation: {:?}", duration);

    // BLAS SGEMM
    if DO_BLAS {
        let time = std::time::Instant::now();
        blas_utils::sgemm(b'N', b'N', m, n, k, 2.0, &a, m, &b, k, 0.0f32, &mut c_ref, m);

        let duration = time.elapsed();
        println!("Time elapsed in BLAS: {:?}", duration);
    }


    // Tiling
    let time = std::time::Instant::now();

    let a = TiledMatrix::<f32>::from(&a, m, k, m);
    let b = TiledMatrix::<f32>::from(&b, k, n, k);

    let duration = time.elapsed();
    println!("Time elapsed in tiling: {:?}", duration);


    // GEMM
    let time = std::time::Instant::now();

    let c = TiledMatrix::<f32>::gemm(2.0, &a, &b);
    let duration = time.elapsed();

    println!("Time elapsed in sgemm: {:?}", duration);


    // GEMM
    let time = std::time::Instant::now();

    let c_gpu = TiledMatrix::<f32>::gemm_gpu(2.0, &a, &b);
    let duration = time.elapsed();

    println!("Time elapsed in sgemm gpu: {:?}", duration);


    // Untiling
    let mut c_vec = vec![ 0. ; m*n ];
    let time = std::time::Instant::now();

    c.copy_in_vec(&mut c_vec, m);

    let duration = time.elapsed();
    println!("Time elapsed in untiling: {:?}", duration);

    if DO_BLAS {
        assert_eq!(c_vec, c_ref);
        let mut c_vec_gpu = vec![ 0. ; m*n ];
        c_gpu.copy_in_vec(&mut c_vec_gpu, m);
        assert!(c_vec == c_vec_gpu);
    }

}


