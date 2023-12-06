use std::iter::zip;

use crate::blas_utils;

/// A constant representing the leading dimension of arrays in tiles,
/// which is also the maximum number of rows and columns a `Tile` can
/// have.
pub const TILE_SIZE: usize = 1024;

/// A `Tile` is a data structure that represents a dense block of a
/// matrix, often used in block matrix operations to optimize for
/// cache usage, parallelism, and memory bandwidth.
///
/// Generic over `T` which is the type of elements stored in the
/// tile.
#[derive(Debug,PartialEq,Clone)]
pub struct Tile<T>
{
    /// A flat vector that contains the elements of the tile. The
    /// elements are stored in column-major order.
    pub(crate) data: Vec<T>,

    /// The number of rows in the tile, not exceeding `TILE_SIZE`.
    pub(crate) nrows: usize,

    /// The number of columns in the tile, not exceeding `TILE_SIZE`.
    pub(crate) ncols: usize,

    /// Flag to specify if the matrix is transposed. For transposed
    /// matrices, the terms row and column need to be swapped.
    pub(crate) transposed: bool,

}


/// # Tile
macro_rules! impl_tile {
    ($s:ty, $gemm:path) => {
        impl Tile<$s>
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
            pub fn new(nrows: usize, ncols: usize, init: $s) -> Self {
                assert!(nrows <= TILE_SIZE, "Too many rows: {nrows} > {TILE_SIZE}");
                assert!(ncols <= TILE_SIZE, "Too many columns: {ncols} > {TILE_SIZE}");
                let size = ncols * nrows;
                let data = vec![init ; size];
                Self { data, nrows, ncols, transposed: false }
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
            pub fn from(other: &[$s], nrows: usize, ncols:usize, lda:usize) -> Self {
                assert!(nrows <= TILE_SIZE, "Too many rows: {nrows} > {TILE_SIZE}");
                assert!(ncols <= TILE_SIZE, "Too many columns: {ncols} > {TILE_SIZE}");
                let size = ncols * nrows;
                let mut data = Vec::<$s>::with_capacity(size);
                for j in 0..ncols {
                    for i in 0..nrows {
                        data.push(other[i + j*lda]);
                    }
                }
                Self { data, nrows, ncols, transposed: false }
            }


            /// Copies the tile's data into a provided mutable slice,
            /// effectively transforming the tile's internal one-dimensional
            /// representation back into a two-dimensional array layout.
            ///
            /// # Arguments
            ///
            /// * `other` - The mutable slice into which the tile's data will be copied.
            /// * `lda` - The leading dimension to be used when copying the data into `other`.
            pub fn copy_in_vec(&self, other: &mut [$s], lda:usize) {
                match self.transposed {
                    false => {
                        for j in 0..self.ncols {
                            let shift_tile = j*self.nrows;
                            let shift_array = j*lda;
                            other[shift_array..(self.nrows + shift_array)].copy_from_slice(&self.data[shift_tile..(self.nrows + shift_tile)]);
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
            #[inline]
            pub unsafe fn get_unchecked(&self, i:usize, j:usize) -> &$s {
                match self.transposed {
                    false => unsafe { self.data.get_unchecked(i + j * self.nrows) },
                    true  => unsafe { self.data.get_unchecked(j + i * self.nrows) },
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
            #[inline]
            pub unsafe fn get_unchecked_mut(&mut self, i:usize, j:usize) -> &mut $s {
                match self.transposed {
                    false => unsafe { self.data.get_unchecked_mut(i + j * self.nrows) },
                    true  => unsafe { self.data.get_unchecked_mut(j + i * self.nrows) },
                }
            }

            /// Returns the number of rows in the tile.
            #[inline]
            pub fn nrows(&self) -> usize {
                match self.transposed {
                    false => self.nrows,
                    true  => self.ncols,
                }
            }

            /// Tells if the tile is transposed or not.
            #[inline]
            pub fn transposed(&self) -> bool {
                self.transposed
            }

            /// Returns the number of columns in the tile.
            #[inline]
            pub fn ncols(&self) -> usize {
                match self.transposed {
                    false => self.ncols,
                    true  => self.nrows,
                }
            }

            /// Transposes the current tile
            #[inline]
            pub fn transpose_mut(&mut self) {
                self.transposed = ! self.transposed;
            }


            /// Returns the transposed of the current tile
            #[inline]
            pub fn t(&self) -> Self {
                Self {
                    transposed: !self.transposed,
                    data: self.data.clone(),
                    ..(*self)
                }
            }

            /// Rescale the tile
            #[inline]
            pub fn scale_mut(&mut self, factor: $s) {
                for x in &mut self.data {
                    *x = *x * factor;
                }
            }

            /// Returns a copy of the tile, rescaled
            #[inline]
            pub fn scale(&self, factor: $s) -> Self {
                let mut result = self.clone();
                for x in &mut result.data {
                    *x = *x * factor;
                }
                result
            }

            /// Add another tile to the tile
            #[inline]
            pub fn add_mut(&mut self, other: &Self) {
                assert_eq!(self.ncols(), other.ncols());
                assert_eq!(self.nrows(), other.nrows());
                for j in 0..self.ncols() {
                  for i in 0..self.nrows() {
                    self[(i,j)] += other[(i,j)];
                  }
                }
            }

            /// Adds another tile to the tile and returns a new tile with the result.
            #[inline]
            pub fn add(&self, other: &Self) -> Self {
                let mut result = self.clone();
                result.add_mut(other);
                result
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
            pub fn geam (alpha: $s, a: &Self, beta: $s, b: &Self) -> Self
            {
                assert_eq!(a.nrows(), b.nrows());
                assert_eq!(a.ncols(), b.ncols());
                let nrows = a.nrows;
                let ncols = a.ncols;
                let size = ncols * nrows;
                let mut data = Vec::<$s>::with_capacity(size);
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
                        for _ in 0..size {
                            data.push( 0.0 );
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
                        for j in 0..(a.ncols()) {
                            for i in 0..(a.nrows()) {
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
            pub fn geam_mut (alpha: $s, a: &Self, beta: $s, b: &Self, c: &mut Self)
            {
                assert_eq!(a.ncols(), b.ncols());
                assert_eq!(a.ncols(), c.ncols());
                assert_eq!(a.nrows(), b.nrows());
                assert_eq!(a.nrows(), c.nrows());
                assert!(!c.transposed);

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
                        for x in &mut c.data[..] {
                            *x = 0.0;
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


            /// Performs a BLAS GEMM operation using `Tiles` $A$, $B$ and $C:
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
            pub fn gemm_mut (alpha: $s, a: &Self, b: &Self, beta: $s, c: &mut Self)
            {
                assert_eq!(a.ncols(), b.nrows());
                assert_eq!(a.nrows(), c.nrows());
                assert_eq!(b.ncols(), c.ncols());
                assert!(!c.transposed);

                let lda = a.nrows;
                let ldb = b.nrows;
                let ldc = c.nrows;

                let m = a.nrows();
                let n = b.ncols();
                let k = a.ncols();

                let transa = if a.transposed { b'T' } else { b'N' };
                let transb = if b.transposed { b'T' } else { b'N' };

                $gemm(transa, transb, m, n, k, alpha, &a.data, lda, &b.data, ldb, beta, &mut c.data, ldc);

            }


            /// Generates a new `Tile` $C$ which is the result of a BLAS GEMM
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
            pub fn gemm (alpha: $s, a: &Self, b: &Self) -> Self
            {
                let mut c = Self::new(a.nrows(), b.ncols(), 0.0);
                Self::gemm_mut(alpha, a, b, 0.0, &mut c);
                c
            }

        }


        /// Implementation of the Index trait to allow for read access to
        /// elements in the Tile using array indexing syntax.
        impl std::ops::Index<(usize,usize)> for Tile<$s>
        {
            type Output = $s;
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

            #[inline]
            fn index(&self, (i,j): (usize,usize)) -> &Self::Output {
                match self.transposed {
                    false => { assert!(i < self.nrows && j < self.ncols); &self.data[i + j * self.nrows] },
                    true  => { assert!(j < self.nrows && i < self.ncols); &self.data[j + i * self.nrows] },
                }
            }
        }

        /// Implementation of the IndexMut trait to allow for write access to
        /// elements in the Tile using array indexing syntax.
        impl std::ops::IndexMut<(usize,usize)> for Tile<$s>
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
            #[inline]
            fn index_mut(&mut self, (i,j): (usize,usize)) -> &mut Self::Output {
                let transposed = self.transposed;
                let nrows = self.nrows;
                let ncols = self.ncols;
                match transposed {
                    false => {assert!(i < nrows && j < ncols);
                            &mut self.data[i + j * nrows]},
                    true  => {assert!(j < nrows && i < ncols);
                            &mut self.data[j + i * nrows]},
                }
            }
        }
    }
}



impl_tile!(f32, blas_utils::sgemm);
impl_tile!(f64, blas_utils::dgemm);

// ------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    // Tests for the Tile implementation...

    use super::*;

    macro_rules! impl_tests {
        ($s:ty
        ,$creation:ident
        ,$creation_too_large:ident
        ,$row_overflow:ident
        ,$col_overflow:ident
        ,$index_mut:ident
        ,$transposition:ident
        ,$scale:ident
        ,$geam_wrong_size:ident
        ,$geam_cases:ident
        ,$gemm:ident
         ) => {

            #[test]
            fn $creation() {
                let tile = Tile::<$s>::new(10, 20, 1.0);
                assert_eq!(tile.nrows(), 10);
                assert_eq!(tile.ncols(), 20);
                for j in 0..(tile.ncols()) {
                    for i in 0..(tile.nrows()) {
                        assert_eq!(tile[(i,j)], 1.0);
                    }
                }

                let mut other_ref = vec![0.0 ; 150];
                let mut other = Vec::<$s>::with_capacity(300);
                for j in 0..15 {
                    for i in 0..10 {
                        let x =  (i as $s) + (j as $s)*100.0;
                        other.push(x);
                        other_ref[i + j*10] = x;
                    }
                    for _ in 0..10 {
                        other.push( 0. );
                    }
                }
                let tile = Tile::<$s>::from(&other, 10, 15, 20);

                let mut other = vec![0.0 ; 150];
                tile.copy_in_vec(&mut other, 10);
                assert_eq!(other, other_ref);
            }

            #[test]
            #[should_panic]
            fn $creation_too_large() {
                let _ = Tile::<$s>::new(TILE_SIZE+1, 10, 1.0);
            }

            #[test]
            #[should_panic]
            fn $row_overflow() {
                let tile = Tile::<$s>::new(10, 20, 1.0);
                let _ = tile[(11,10)];
            }

            #[test]
            #[should_panic]
            fn $col_overflow() {
                let tile = Tile::<$s>::new(10, 20, 1.0);
                let _ = tile[(1,21)];
            }

            #[test]
            fn $index_mut() {
                let mut tile = Tile::<$s>::new(10, 20, 1.0);
                for i in 0..10 {
                    for j in 0..20 {
                        tile[(i,j)] = (i as $s) * 100.0 + (j as $s);
                    }
                }
                let mut ref_val = vec![0. ; 20*10];
                for j in 0..20 {
                    for i in 0..10 {
                        ref_val[i + j*10] = (i as $s) * 100.0 + (j as $s);
                    }
                }
                assert_eq!(tile.data, ref_val);

            }

            #[test]
            fn $transposition() {
                let mut tile = Tile::<$s>::new(10, 20, 1.0);
                for i in 0..10 {
                    for j in 0..20 {
                        tile[(i,j)] = (i as $s) * 100.0 + (j as $s);
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
            fn $scale() {
                let mut tile = Tile::<$s>::new(10, 20, 1.0);
                for i in 0..10 {
                    for j in 0..20 {
                        tile[(i,j)] = (i as $s) * 100.0 + (j as $s);
                    }
                }
                let mut data_ref = Vec::from(&tile.data[..]);
                for i in 0..10 {
                    for j in 0..20 {
                        data_ref[i+j*10] *= 2.0;
                    }
                }
                let mut new_tile = tile.scale(2.0);
                assert_eq!(new_tile.data, data_ref);

                tile.scale_mut(2.0);
                assert_eq!(tile.data, data_ref);
            }

            #[test]
            #[should_panic]
            fn $geam_wrong_size() {
                let n = 10;
                let m = 5;
                let a = Tile::<$s>::new(m, n, 0.0);
                let b = Tile::<$s>::new(n, m, 0.0);
                let _ = Tile::<$s>::geam(1.0, &a, 1.0, &b);
            }

            #[test]
            fn $geam_cases() {
                let n = 8;
                let m = 4;
                let mut a   = Tile::<$s>::new(m, n, 0.0);
                let mut a_t = Tile::<$s>::new(n, m, 0.0);
                let mut b   = Tile::<$s>::new(m, n, 0.0);
                let mut b_t = Tile::<$s>::new(n, m, 0.0);
                let zero_tile = Tile::<$s>::new(m, n, 0.0);
                let zero_tile_t = Tile::<$s>::new(n, m, 0.0);
                for i in 0..m {
                    for j in 0..n {
                        a[(i,j)] = (i * 10 + j) as $s;
                        b[(i,j)] = (i * 1000 + j*10) as $s;
                        a_t[(j,i)] = (i * 10 + j) as $s;
                        b_t[(j,i)] = (i * 1000 + j*10) as $s;
                    }
                }

                assert_eq!(
                    Tile::<$s>::geam(0.0, &a, 0.0, &b),
                    zero_tile);

                assert_eq!(
                    Tile::<$s>::geam(1.0, &a, 0.0, &b),
                    a);

                assert_eq!(
                    Tile::<$s>::geam(0.0, &a, 1.0, &b),
                    b);

                assert_eq!(
                    Tile::<$s>::geam(2.0, &a, 0.0, &b),
                    { let mut r = Tile::<$s>::new(m, n, 0.0);
                    for i in 0..m {
                        for j in 0..n {
                            r[(i,j)] = 2.0*a[(i,j)];
                        }
                    };
                    r });

                assert_eq!(
                    Tile::<$s>::geam(0.5, &a, 0.5, &a),
                    a);

                assert_eq!(
                    Tile::<$s>::geam(0.5, &a, -0.5, &a),
                    zero_tile);

                assert_eq!(
                    Tile::<$s>::geam(1.0, &a, -1.0, &a),
                    zero_tile);

                assert_eq!(
                    Tile::<$s>::geam(0.0, &a, 2.0, &b),
                    { let mut r = Tile::<$s>::new(m, n, 0.0);
                    for i in 0..m {
                        for j in 0..n {
                            r[(i,j)] = 2.0*b[(i,j)];
                        }
                    };
                    r });

                assert_eq!(
                    Tile::<$s>::geam(1.0, &a, 1.0, &b),
                    { let mut r = Tile::<$s>::new(m, n, 0.0);
                    for i in 0..m {
                        for j in 0..n {
                            r[(i,j)] = a[(i,j)] + b[(i,j)];
                        }
                    };
                    r });

                assert_eq!(
                    Tile::<$s>::geam(-1.0, &a, 1.0, &b),
                    { let mut r = Tile::<$s>::new(m, n, 0.0);
                    for i in 0..m {
                        for j in 0..n {
                            r[(i,j)] = b[(i,j)] - a[(i,j)];
                        }
                    };
                    r });

                assert_eq!(
                    Tile::<$s>::geam(1.0, &a, -1.0, &b),
                    { let mut r = Tile::<$s>::new(m, n, 0.0);
                    for i in 0..m {
                        for j in 0..n {
                            r[(i,j)] = a[(i,j)] - b[(i,j)];
                        }
                    };
                    r });

                assert_eq!(
                    Tile::<$s>::geam(1.0, &a, -1.0, &a_t.t()),
                    zero_tile);

                assert_eq!(
                    Tile::<$s>::geam(1.0, &a_t.t(), -1.0, &a),
                    zero_tile_t);


                // Mutable geam

                assert_eq!(
                    { let mut c = Tile::<$s>::geam(1.0, &a, 1.0, &b);
                    Tile::<$s>::geam_mut(-1.0, &a, 1.0, &(c.clone()), &mut c);
                    c} ,
                    b);

                for (alpha, beta) in [ (1.0,1.0), (1.0,-1.0), (-1.0,-1.0), (-1.0,1.0),
                                        (1.0,0.0), (0.0,-1.0), (0.5, 1.0), (0.5, 1.0),
                                        (0.5,-0.5) ] {
                    assert_eq!(
                        { let mut c = a.clone();
                        Tile::<$s>::geam_mut(alpha, &a, beta, &b, &mut c);
                        c},
                        Tile::<$s>::geam(alpha, &a, beta, &b));
                };
            }

            #[test]
            fn $gemm() {
                let a = Tile::<$s>::from( &[1. , 1.1, 1.2, 1.3,
                                    2. , 2.1, 2.2, 2.3,
                                    3. , 3.1, 3.2, 3.3],
                                    4, 3, 4).t();

                let b = Tile::<$s>::from( &[ 1.0, 2.0, 3.0, 4.0,
                                    1.1, 2.1, 3.1, 4.1],
                                    4, 2, 4 );

                let c_ref = Tile::<$s>::from( &[12.0 , 22.0 , 32.0,
                                        12.46, 22.86, 33.26],
                                        3, 2, 3);

                let c = Tile::<$s>::gemm(1.0, &a, &b);
                let difference = Tile::<$s>::geam(1.0, &c, -1.0, &c_ref);
                for j in 0..2 {
                    for i in 0..3 {
                        assert!(num::abs(difference[(i,j)] / c[(i,j)]) < <$s>::EPSILON);
                    }
                }

                let a = a.t();
                let b = b.t();
                let c_t = Tile::<$s>::gemm(1.0, &b, &a);
                let difference = Tile::<$s>::geam(1.0, &c_t, -1.0, &c.t());
                for j in 0..3 {
                    for i in 0..2 {
                        assert_eq!(difference[(i,j)], 0.0);
                    }
                }
            }

        }
    }

    impl_tests!(    f32,
                    creation_32,
                    creation_too_large_32,
                    row_overflow_32,
                    col_overflow_32,
                    index_mut_32,
                    transposition_32,
                    scale_32,
                    geam_wrong_size_32,
                    geam_cases_32,
                    gemm_32);

    impl_tests!(    f64,
                    creation_64,
                    creation_too_large_64,
                    row_overflow_64,
                    col_overflow_64,
                    index_mut_64,
                    transposition_64,
                    scale_64,
                    geam_wrong_size_64,
                    geam_cases_64,
                    gemm_64);
}



