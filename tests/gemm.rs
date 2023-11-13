use samo::TiledMatrix;
use samo::tiled_matrix;

mod helper_blas;
use crate::helper_blas::{blas_dgemm, blas_sgemm};

#[test]
#[ignore]
pub fn time_dgemm() {
    let m = 10100;
    let n = 20200;
    let k = 3030;

    let time = std::time::Instant::now();

    let mut a = vec![ 0. ; m*k ];
    for j in 0..k {
        for i in 0..m {
            a[i + j*m] = (i as f64) + (j as f64)*10.0;
        }
    }

    let mut b = vec![ 0. ; k*n ];
    for j in 0..n {
        for i in 0..k {
            b[i + j*k] = -(i as f64) + (j as f64)*7.0;
        }
    }

    let mut c_ref = vec![ 1. ; m*n ];

    let duration = time.elapsed();
    println!("Time elapsed in preparation: {:?}", duration);

    let time = std::time::Instant::now();
    blas_dgemm(b'N', b'N', m, n, k, 2.0, &a, m, &b, k, 0.0f64, &mut c_ref, m);

    let duration = time.elapsed();
    println!("Time elapsed in BLAS: {:?}", duration);


    let time = std::time::Instant::now();

    let a = TiledMatrix::from(&a, m, k, m);
    let b = TiledMatrix::from(&b, k, n, k);

    let duration = time.elapsed();
    println!("Time elapsed in tiling: {:?}", duration);


    let time = std::time::Instant::now();

    let c = tiled_matrix::gemm(2.0, &a, &b);
    let duration = time.elapsed();

    println!("{}", c[(0,0)]);
    println!("Time elapsed in dgemm: {:?}", duration);
    assert!(false);

}

#[test]
#[ignore]
pub fn time_sgemm() {
    let m = 10100;
    let n = 20200;
    let k = 3030;

    let time = std::time::Instant::now();

    let mut a = vec![ 0. ; m*k ];
    for j in 0..k {
        for i in 0..m {
            a[i + j*m] = (i as f32) + (j as f32)*10.0;
        }
    }

    let mut b = vec![ 0. ; k*n ];
    for j in 0..n {
        for i in 0..k {
            b[i + j*k] = -(i as f32) + (j as f32)*7.0;
        }
    }

    let mut c_ref = vec![ 1. ; m*n ];

    let duration = time.elapsed();
    println!("Time elapsed in preparation: {:?}", duration);

    let time = std::time::Instant::now();
//    blas_sgemm(b'N', b'N', m, n, k, 2.0, &a, m, &b, k, 0.0f32, &mut c_ref, m);

    let duration = time.elapsed();
    println!("Time elapsed in BLAS: {:?}", duration);


    let time = std::time::Instant::now();

    let a = TiledMatrix::from(&a, m, k, m);
    let b = TiledMatrix::from(&b, k, n, k);

    let duration = time.elapsed();
    println!("Time elapsed in tiling: {:?}", duration);


    let time = std::time::Instant::now();

    let c = tiled_matrix::gemm(2.0, &a, &b);
    let duration = time.elapsed();

    println!("{}", c[(0,0)]);
    println!("Time elapsed in sgemm: {:?}", duration);
    assert!(false);

}


