extern crate blas;
extern crate intel_mkl_src;

use std::iter::zip;
use num::traits::Float;

/// A constant representing the leading dimension of arrays in tiles,
/// which is also the maximum number of rows and columns a `Tile` can
/// have.
pub const TILE_SIZE: usize = 32;

/// A `Tile` is a data structure that represents a dense block of a
/// matrix, often used in block matrix operations to optimize for
/// cache usage, parallelism, and memory bandwidth.
/// 
/// Generic over `T` which is the type of elements stored in the
/// tile. It requires `T` to have the `Float` trait.
#[derive(Debug)]
#[derive(PartialEq)]
pub struct Tile<T>
where
    T: Float
{
    /// A flat vector that contains the elements of the tile. The
    /// elements are stored in column-major order.
    data: Vec<T>,

    /// The number of rows in the tile, not exceeding `TILE_SIZE`.
    nrows: usize,

    /// The number of columns in the tile, not exceeding `TILE_SIZE`.
    ncols: usize,

    /// Flag to specify if the matrix is transposed. For transposed
    /// matrices, the terms row and column need to be swapped.
    transposed: bool,
}


impl<T> Tile<T>
where
    T: Float
{
    /// Constructs a new `Tile` with the specified number of rows and
    /// columns, initializing all elements to the provided `init`
    /// value.
    /// 
    /// # Arguments
    /// 
    /// * `nrows` - The number of rows for the tile.
    /// * `ncols` - The number of columns for the tile.
    /// * `init` - Initial value to set all elements of the tile.
    /// 
    /// # Panics
    /// 
    /// Panics if `nrows` or `ncols` exceed `TILE_SIZE`, which is the
    /// maximum allowed dimension.
    pub fn new(nrows: usize, ncols: usize, init: T) -> Self {
        if nrows > TILE_SIZE {panic!("Too many rows");}
        if ncols > TILE_SIZE {panic!("Too many columns");}
        let size = ncols * nrows;
        let mut data = Vec::<T>::with_capacity(size);
        for _ in 0..ncols {
            for _ in 0..nrows {
                data.push(init);
            }
        }
        Tile { data, nrows, ncols, transposed: false }
    }

    /// Constructs a `Tile` from a slice of data, given the number of
    /// rows and columns and the leading dimension (`lda`) of the
    /// original data.
    /// 
    /// # Arguments
    /// 
    /// * `other` - The slice from which to construct the tile.
    /// * `nrows` - The number of rows in the tile.
    /// * `ncols` - The number of columns in the tile.
    /// * `lda` - The leading dimension of the data in `other`.
    /// 
    /// # Panics
    /// 
    /// Panics if `nrows` or `ncols` exceed `TILE_SIZE`.
    pub fn from(other: &[T], nrows: usize, ncols:usize, lda:usize) -> Self {
        if nrows > TILE_SIZE {panic!("Too many rows");}
        if ncols > TILE_SIZE {panic!("Too many columns");}
        let size = ncols * nrows;
        let mut data = Vec::<T>::with_capacity(size);
        for j in 0..ncols {
            for i in 0..nrows {
                data.push(other[i + j*lda]);
            }
        }
        Tile { data, nrows, ncols, transposed: false }
    }


    /// Copies the tile's data into a provided mutable slice,
    /// effectively transforming the tile's internal one-dimensional
    /// representation back into a two-dimensional array layout.
    /// 
    /// # Arguments
    /// 
    /// * `other` - The mutable slice into which the tile's data will be copied.
    /// * `lda` - The leading dimension to be used when copying the data into `other`.
    pub fn copy_in_vec(&self, other: &mut [T], lda:usize) {
        match self.transposed {
            false => {
                for j in 0..self.ncols {
                    let shift_tile = j*self.nrows;
                    let shift_array = j*lda;
                    for i in 0..self.nrows {
                        other[i + shift_array] = self.data[i + shift_tile];
                    }
                }},
            true => {
                for i in 0..self.nrows {
                    let shift_array = i*lda;
                    for j in 0..self.ncols {
                        let shift_tile = j*self.nrows;
                        other[j + shift_array] = self.data[i + shift_tile];
                    }
                }},
        }
    }

    /// Returns a reference to the element at the specified (row, column) index,
    /// without bounds checking.
    /// 
    /// # Arguments
    /// 
    /// * `i` - The row index of the element.
    /// * `j` - The column index of the element.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is
    /// /undefined behavior/ even if the resulting reference is not used.
    pub unsafe fn get_unchecked(&self, i:usize, j:usize) -> &T {
        match self.transposed {
            false => unsafe { self.data.get_unchecked(i + j * &self.nrows) },
            true  => unsafe { self.data.get_unchecked(j + i * &self.nrows) },
        }
    }

    /// Returns a mutable reference to element at the specified (row, column) index,
    /// without bounds checking.
    /// 
    /// # Arguments
    /// 
    /// * `i` - The row index of the element.
    /// * `j` - The column index of the element.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is
    /// /undefined behavior/ even if the resulting reference is not used.
    pub unsafe fn get_unchecked_mut(&mut self, i:usize, j:usize) -> &mut T {
        match self.transposed {
            false => unsafe { self.data.get_unchecked_mut(i + j * &self.nrows) },
            true  => unsafe { self.data.get_unchecked_mut(j + i * &self.nrows) },
        }
    }

    /// Returns the number of rows in the tile.
    pub fn nrows(&self) -> usize {
        match self.transposed {
            false => self.nrows, 
            true  => self.ncols,
        }
    }

    /// Tells if the tile is transposed or not.
    pub fn transposed(&self) -> bool {
        self.transposed
    }

    /// Returns the number of columns in the tile.
    pub fn ncols(&self) -> usize {
        match self.transposed {
            false => self.ncols, 
            true  => self.nrows,
        }
    }

    /// Transposes the current tile
    pub fn transpose_mut(&mut self) {
        self.transposed = ! self.transposed;
    }


    /// Returns the transposed of the current tile
    pub fn t(&self) -> Self {
        Tile {
            transposed: !self.transposed,
            data: self.data.clone(),
            ..(*self)
        }
    }
}

/// Implementation of the Index trait to allow for read access to
/// elements in the Tile using array indexing syntax.
impl<T> std::ops::Index<(usize,usize)> for Tile<T>
where
    T: Float
{
     type Output = T;
    /// Returns a reference to the element at the given (row, column)
    /// index, using the column-major order.
    /// 
    /// # Arguments
    /// 
    /// * `(i, j)` - A tuple containing the row and column index for the element.
    /// 
    /// # Panics
    /// 
    /// Panics if the specified indices are out of bounds.

     fn index(&self, (i,j): (usize,usize)) -> &Self::Output {
        match self.transposed {
            false => {assert!(i < self.nrows && j < self.ncols);
                      &self.data[i + j * &self.nrows]},
            true  => {assert!(j < self.nrows && i < self.ncols);
                      &self.data[j + i * &self.nrows]},
        }
     }
}

/// Implementation of the IndexMut trait to allow for write access to
/// elements in the Tile using array indexing syntax.
impl<T> std::ops::IndexMut<(usize,usize)> for Tile<T>
where
    T: Float
{
    /// Returns a mutable reference to the element at the given (row,
    /// column) index, using the row-major order.
    ///
    /// # Arguments
    ///
    /// * (i, j) - A tuple containing the row and column index for the element.
    ///
    /// # Panics
    ///
    /// Panics if the specified indices are out of bounds.
    fn index_mut(&mut self, (i,j): (usize,usize)) -> &mut Self::Output {
        match self.transposed {
            false => {assert!(i < self.nrows && j < self.ncols);
                      &mut self.data[i + j * &self.nrows]},
            true  => {assert!(j < self.nrows && i < self.ncols);
                      &mut self.data[j + i * &self.nrows]},
        }
    }
}

/// Combines two `Tile`s $A$ and $B$ with coefficients $\alpha$ and
/// $\beta$, and returns a new `Tile` $C$:
/// $$C = \alpha A + \beta B$$.
/// 
/// # Arguments
/// 
/// * `alpha` - $\alpha$
/// * `a` - Tile $A$
/// * `beta` - $\beta$
/// * `b` - Tile $B$
/// 
/// # Panics
/// 
/// Panics if the tiles don't have the same size.
pub fn geam<T> (alpha: T, a: &Tile<T>, beta: T, b: &Tile<T>) -> Tile<T>
where T: Float {
    assert!(a.nrows() == b.nrows() && a.ncols() == b.ncols());
    let nrows = a.nrows;
    let ncols = a.ncols;
    let size = ncols * nrows;
    let mut data = Vec::<T>::with_capacity(size);
    let mut transposed = false;

    let make_pattern = |x| {
        if x == T::zero() { 0 }
        else if x == T::one() { 1 }
        else if x == -T::one() { -1 }
        else { 2 }
    };
    
    let _a = make_pattern(alpha);
    let _b = make_pattern(beta);

    match (_a, _b, a.transposed, b.transposed) {
        (0,0,_,_) =>  {
            for _ in 0..size {
                data.push( T::zero() );
            }
        },

        (1,0,_,_) =>  {
            transposed = a.transposed;
            for &x in &a.data {
                data.push( x );
            }
        },

        (-1,0,_,_) =>  {
            transposed = a.transposed;
            for &x in &a.data {
                data.push( -x );
            }
        },

        (_,0,_,_) =>  {
            transposed = a.transposed;
            for &x in &a.data {
                data.push( alpha * x );
            }
        },

        (0,1,_,_) =>  {
            transposed = b.transposed;
            for &x in &b.data {
                data.push( x );
            }
        },

        (0,-1,_,_) =>  {
            transposed = b.transposed;
            for &x in &b.data {
                data.push( -x );
            }
        },

        (0,_,_,_) =>  {
            transposed = b.transposed;
            for &x in &b.data {
                data.push( beta * x );
            }
        },

        (1, 1, false, false) | (1, 1, true, true) => {
            transposed = a.transposed;
            for (&x,&y) in zip(&a.data, &b.data) {
                data.push( x + y );
            }},

        (1,-1, false, false) | (1,-1, true, true) => {
            transposed = a.transposed;
            for (&x,&y) in zip(&a.data, &b.data) {
                data.push( x - y );
            }},

        (_, _, false, false) | (_, _, true, true) => {
            transposed = a.transposed;
            for (&x,&y) in zip(&a.data, &b.data) {
                data.push( alpha * x + beta * y );
            }},
        
        _ => {
            for j in 0..ncols {
                for i in 0..nrows {
                    let x = a[(i,j)];
                    let y = b[(i,j)];
                    data.push( alpha * x + beta * y );
                }
            }
        },
    };
    Tile { data, nrows, ncols, transposed }
}


/// Combines two `Tile`s $A$ and $B$ with coefficients $\alpha$ and
/// $\beta$, into `Tile` $C$:
/// $$C = \alpha A + \beta B$$.
/// 
/// # Arguments
/// 
/// * `alpha` - $\alpha$
/// * `a` - Tile $A$
/// * `beta` - $\beta$
/// * `b` - Tile $B$
/// * `c` - Tile $C$
/// 
/// # Panics
/// 
/// Panics if the tiles don't have the same size.
pub fn geam_mut<T> (alpha: T, a: &Tile<T>, beta: T, b: &Tile<T>, c: &mut Tile<T>)
where T: Float {
    assert!(a.nrows() == b.nrows() && a.nrows() == c.nrows() &&
            a.ncols() == b.ncols() && a.ncols() == c.ncols);
    let nrows = a.nrows;
    let ncols = a.ncols;
    let mut transposed = false;

    let make_pattern = |x| {
        if x == T::zero() { 0 }
        else if x == T::one() { 1 }
        else if x == -T::one() { -1 }
        else { 2 }
    };
    
    let _a = make_pattern(alpha);
    let _b = make_pattern(beta);

    match (_a, _b, a.transposed, b.transposed) {
        (0,0,_,_) =>  {
            for x in &mut c.data[..] {
                *x = T::zero();
            }
        },

        (1,0,_,_) =>  {
            transposed = a.transposed;
            for (x, &v) in zip(&mut c.data, &a.data) {
                *x = v;
            }
        },

        (-1,0,_,_) =>  {
            transposed = a.transposed;
            for (x, &v) in zip(&mut c.data, &a.data) {
                *x = -v;
            }
        },

        (_,0,_,_) =>  {
            transposed = a.transposed;
            for (x, &v) in zip(&mut c.data, &a.data) {
                *x = alpha*v;
            }
        },

        (0,1,_,_) =>  {
            transposed = b.transposed;
            for (x, &v) in zip(&mut c.data, &b.data) {
                *x = v;
            }
        },

        (0,-1,_,_) =>  {
            transposed = b.transposed;
            for (x, &v) in zip(&mut c.data, &b.data) {
                *x = -v;
            }
        },

        (0,_,_,_) =>  {
            transposed = b.transposed;
            for (x, &v) in zip(&mut c.data, &b.data) {
                *x = -beta*v;
            }
        },

        (1, 1, false, false) | (1, 1, true, true) => {
            transposed = a.transposed;
            for (x, (&v, &w)) in zip(&mut c.data, zip(&a.data, &b.data)) {
                *x = v + w;
            }},

        (1,-1, false, false) | (1,-1, true, true) => {
            transposed = a.transposed;
            for (x, (&v, &w)) in zip(&mut c.data, zip(&a.data, &b.data)) {
                *x = v - w;
            }},

        (_, _, false, false) | (_, _, true, true) => {
            transposed = a.transposed;
            for (x, (&v, &w)) in zip(&mut c.data, zip(&a.data, &b.data)) {
                *x = alpha * v + beta * w;
            }},
        
        _ => {
            for j in 0..ncols {
                for i in 0..nrows {
                    let x = a[(i,j)];
                    let y = b[(i,j)];
                    c[(i,j)] = alpha * x + beta * y;
                }
            }
        },
    };
    c.transposed = transposed;
}


/// Performs a BLAS DGEMM operation using `Tiles` $A$, $B$ and $C:
/// $$C = \alpha A \dot B + \beta C$$.
/// `Tile` $C$ is mutated.
/// 
/// # Arguments
/// 
/// * `alpha` - $\alpha$
/// * `a` - Tile $A$
/// * `b` - Tile $B$
/// * `beta` - $\beta$
/// * `c` - Tile $C$
/// 
/// # Panics
/// 
/// Panics if the tiles don't have matching sizes.
pub fn dgemm_mut (alpha: f64, a: &Tile<f64>, b: &Tile<f64>, beta: f64, c: &mut Tile<f64>) {
    assert!(a.ncols() == b.nrows());
    assert!(a.nrows() == c.nrows());
    assert!(b.ncols() == c.ncols());

    let lda = a.nrows;
    let ldb = b.nrows;
    let ldc = c.nrows;

    let m : i32 = a.nrows().try_into().unwrap();
    let n : i32 = b.ncols().try_into().unwrap();
    let k : i32 = a.ncols().try_into().unwrap();

    let lda : i32 = lda.try_into().unwrap();
    let ldb : i32 = ldb.try_into().unwrap();
    let ldc : i32 = ldc.try_into().unwrap();

    let transa = if a.transposed { b'T' } else { b'N' };
    let transb = if b.transposed { b'T' } else { b'N' };

    unsafe {
        blas::dgemm(transa, transb, m, n, k, alpha, &a.data, lda, &b.data, ldb, beta, &mut c.data, ldc);
    }

}

/// Performs a BLAS SGEMM operation using `Tiles` $A$, $B$ and $C:
/// $$C = \alpha A \dot B + \beta C$$.
/// `Tile` $C$ is mutated.
/// 
/// # Arguments
/// 
/// * `alpha` - $\alpha$
/// * `a` - Tile $A$
/// * `b` - Tile $B$
/// * `beta` - $\beta$
/// * `c` - Tile $C$
/// 
/// # Panics
/// 
/// Panics if the tiles don't have matching sizes.
pub fn sgemm_mut (alpha: f32, a: &Tile<f32>, b: &Tile<f32>, beta: f32, c: &mut Tile<f32>) {
    assert!(a.ncols() == b.nrows());
    assert!(a.nrows() == c.nrows());
    assert!(b.ncols() == c.ncols());

    let lda = a.nrows;
    let ldb = b.nrows;
    let ldc = c.nrows;

    let m : i32 = a.nrows().try_into().unwrap();
    let n : i32 = b.ncols().try_into().unwrap();
    let k : i32 = a.ncols().try_into().unwrap();

    let lda : i32 = lda.try_into().unwrap();
    let ldb : i32 = ldb.try_into().unwrap();
    let ldc : i32 = ldc.try_into().unwrap();

    let transa = if a.transposed { b'T' } else { b'N' };
    let transb = if b.transposed { b'T' } else { b'N' };

    unsafe {
        blas::sgemm(transa, transb, m, n, k, alpha, &a.data, lda, &b.data, ldb, beta, &mut c.data, ldc);
    }

}


/// Generates a new `Tile` $C$ which is the result of a BLAS DGEMM
/// operation between two `Tiles` $A$ and $B$.
/// $$C = \alpha A \dot B$$.
/// 
/// # Arguments
/// 
/// * `alpha` - $\alpha$
/// * `a` - Tile $A$
/// * `b` - Tile $B$
/// 
/// # Panics
/// 
/// Panics if the tiles don't have sizes that match.
pub fn dgemm (alpha: f64, a: &Tile<f64>, b: &Tile<f64>) -> Tile<f64>
{
    let mut c = Tile::new(a.nrows(), b.ncols(), 0.0f64);
    dgemm_mut(alpha, a, b, 0.0f64, &mut c);
    c
}


/// Generates a new `Tile` $C$ which is the result of a BLAS SGEMM
/// operation between two `Tiles` $A$ and $B$.
/// $$C = \alpha A \dot B$$.
/// 
/// # Arguments
/// 
/// * `alpha` - $\alpha$
/// * `a` - Tile $A$
/// * `b` - Tile $B$
/// 
/// # Panics
/// 
/// Panics if the tiles don't have sizes that match.
pub fn sgemm (alpha: f32, a: &Tile<f32>, b: &Tile<f32>) -> Tile<f32>
{
    let mut c = Tile::new(a.nrows(), b.ncols(), 0.0f32);
    sgemm_mut(alpha, a, b, 0.0f32, &mut c);
    c
}




// ------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    // Tests for the Tile implementation...

    use super::*;

    #[test]
    fn creation() {
        let tile = Tile::new(10, 20, 1.0);
        assert_eq!(tile.nrows(), 10);
        assert_eq!(tile.ncols(), 20);
        for j in 0..(tile.ncols()) {
            for i in 0..(tile.nrows()) {
                assert_eq!(tile[(i,j)], 1.0);
            }
        }

        let mut other_ref = vec![0.0 ; 150];
        let mut other = Vec::<f64>::with_capacity(300);
        for j in 0..15 {
            for i in 0..10 {
                let x =  (i as f64) + (j as f64)*100.0;
                other.push(x);
                other_ref[i + j*10] = x;
            }
            for _ in 0..10 {
                other.push( 0. );
            }
        }
        let tile = Tile::from(&other, 10, 15, 20);

        let mut other = vec![0.0 ; 150];
        tile.copy_in_vec(&mut other, 10);
        assert_eq!(other, other_ref);
    }

    #[test]
    #[should_panic]
    fn creation_too_large() {
        let _ = Tile::new(TILE_SIZE+1, 10, 1.0);
    }

    #[test]
    #[should_panic]
    fn row_overflow() {
        let tile = Tile::new(10, 20, 1.0);
        let _ = tile[(11,10)];
    }

    #[test]
    #[should_panic]
    fn col_overflow() {
        let tile = Tile::new(10, 20, 1.0);
        let _ = tile[(1,21)];
    }

    #[test]
    fn index_mut() {
        let mut tile = Tile::new(10, 20, 1.0);
        for i in 0..10 {
            for j in 0..20 {
                tile[(i,j)] = (i as f64) * 100.0 + (j as f64);
            }
        }
        let mut ref_val = vec![0. ; 20*10];
        for j in 0..20 {
            for i in 0..10 {
                ref_val[i + j*10] = (i as f64) * 100.0 + (j as f64);
            }
        }
        assert_eq!(tile.data, ref_val);
        
    }

    #[test]
    fn transposition() {
        let mut tile = Tile::new(10, 20, 1.0);
        for i in 0..10 {
            for j in 0..20 {
                tile[(i,j)] = (i as f64) * 100.0 + (j as f64);
            }
        }
        assert_eq!(tile.nrows(), 10);
        assert_eq!(tile.ncols(), 20);
        assert_eq!(tile[(3,8)],  308.);
        assert_eq!(tile.transposed(), false);

        tile.transpose_mut();
        assert_eq!(tile.nrows(), 20);
        assert_eq!(tile.ncols(), 10);
        assert_eq!(tile[(8,3)],  308.);
        assert_eq!(tile.transposed(), true);
    }

    #[test]
    #[should_panic]
    fn geam_wrong_size() {
        let n = 10;
        let m = 5;
        let a = Tile::new(m, n, 0.0);
        let b = Tile::new(n, m, 0.0);
        let _ = geam(1.0, &a, 1.0, &b);
    }
    
    fn geam_cases() {
        let n = 8;
        let m = 4;
        let mut a   = Tile::new(m, n, 0.0f64);
        let mut a_t = Tile::new(n, m, 0.0f64);
        let mut b   = Tile::new(m, n, 0.0f64);
        let mut b_t = Tile::new(n, m, 0.0f64);
        let zero_tile = Tile::new(m, n, 0.0f64);
        for i in 0..n {
            for j in 0..m {
                a[(i,j)] = (i * 10 + j) as f64;
                b[(i,j)] = (i * 1000 + j*10) as f64;
                a_t[(j,i)] = (i * 10 + j) as f64;
                b_t[(j,i)] = (i * 1000 + j*10) as f64;
            }
        }

        assert_eq!(
            geam(0.0, &a, 0.0, &b),
            zero_tile);

        assert_eq!(
            geam(1.0, &a, 0.0, &b),
            a);

        assert_eq!(
            geam(0.0, &a, 1.0, &b),
            b);

        assert_eq!(
            geam(2.0, &a, 0.0, &b),
            { let mut r = Tile::new(m, n, 0.0f64);
              for i in 0..n {
                  for j in 0..m {
                      r[(i,j)] = 2.0*a[(i,j)];
                  }
              };
              r });

        assert_eq!(
            geam(0.5, &a, 0.5, &a),
            a);

        assert_eq!(
            geam(0.5, &a, -0.5, &a),
            zero_tile);

        assert_eq!(
            geam(1.0, &a, -1.0, &a),
            zero_tile);

        assert_eq!(
            geam(0.0, &a, 2.0, &b),
            { let mut r = Tile::new(m, n, 0.0f64);
              for i in 0..n {
                  for j in 0..m {
                      r[(i,j)] = 2.0*b[(i,j)];
                  }
              };
              r });

        assert_eq!(
            geam(1.0, &a, 1.0, &b),
            { let mut r = Tile::new(m, n, 0.0f64);
              for i in 0..n {
                  for j in 0..m {
                      r[(i,j)] = a[(i,j)] + b[(i,j)];
                  }
              };
              r });

        assert_eq!(
            geam(-1.0, &a, 1.0, &b),
            { let mut r = Tile::new(m, n, 0.0f64);
              for i in 0..n {
                  for j in 0..m {
                      r[(i,j)] = b[(i,j)] - a[(i,j)];
                  }
              };
              r });

        assert_eq!(
            geam(1.0, &a, -1.0, &b),
            { let mut r = Tile::new(m, n, 0.0f64);
              for i in 0..n {
                  for j in 0..m {
                      r[(i,j)] = a[(i,j)] - b[(i,j)];
                  }
              };
              r });

        assert_eq!(
            geam(1.0, &a, -1.0, &a_t.t()),
            zero_tile);

        assert_eq!(
            geam(1.0, &a.t(), -1.0, &a),
            zero_tile);


        // Mutable geam
        
        assert_eq!(
            { let mut c = geam(1.0, &a, 1.0, &b);
              geam_mut(1.0, &a, -1.0, &b, &mut c);
              c} ,
              a);

        for (alpha, beta) in [ (1.0,1.0), (1.0,-1.0), (-1.0,-1.0), (-1.0,1.0),
                                (1.0,0.0), (0.0,-1.0), (0.5, 1.0), (0.5, 1.0),
                                (0.5,-0.5) ] {
            assert_eq!(
                { let mut c = geam(alpha, &a, beta, &b);
                  geam_mut(alpha, &a, beta, &b, &mut c);
                  c},
                geam(2.0*alpha, &a, 2.0*beta, &b));
        };
    }

    fn test_dgemm() {
        let a = Tile::from( &[1. , 1.1, 1.2, 1.3,
                              2. , 2.1, 2.2, 2.3,
                              3. , 3.1, 3.2, 3.3f64],
                              4, 3, 4);

        let b = Tile::from( &[ 1.0, 2.0, 3.0, 4.0,
                               1.1, 2.1, 3.1, 4.1f64],
                               4, 2, 4 );

        let c_ref = Tile::from( &[12.0 , 22.0 , 32.0,
                                  12.46, 22.86, 33.26f64],
                                  3, 2, 3);

        let c = dgemm(1.0, &a, &b);
        assert_eq!(c, c_ref);

        let a = a.t();
        let b = b.t();
        let c = dgemm(1.0, &b, &a);
        assert_eq!(c, c_ref.t());
    }

    fn test_sgemm() {
        let a = Tile::from( &[1. , 1.1, 1.2, 1.3,
                              2. , 2.1, 2.2, 2.3,
                              3. , 3.1, 3.2, 3.3f32],
                              4, 3, 4);

        let b = Tile::from( &[ 1.0, 2.0, 3.0, 4.0,
                               1.1, 2.1, 3.1, 4.1f32],
                               4, 2, 4 );

        let c_ref = Tile::from( &[12.0 , 22.0 , 32.0,
                                  12.46, 22.86, 33.26f32],
                                  3, 2, 3);

        let c = sgemm(1.0, &a, &b);
        assert_eq!(c, c_ref);

        let a = a.t();
        let b = b.t();
        let c = sgemm(1.0, &b, &a);
        assert_eq!(c, c_ref.t());
    }
}
