use crate::cuda;
use crate::cublas;
use crate::tile::Tile;
use crate::tile::TILE_SIZE;

use crate::cuda::DevPtr;

/// A `TileGPU` is a data structure that represents a dense block of a
/// matrix, often used in block matrix operations to optimize for
/// cache usage, parallelism, and memory bandwidth.
///
/// Generic over `T` which is the type of elements stored in the
/// tile.
#[derive(Debug)]
pub struct TileGPU<T>
{
    /// A device pointer to the matrix stored on GPU
    pub(crate) dev_ptr: DevPtr<T>,

    /// The number of rows in the tile, not exceeding `TILE_SIZE`.
    pub(crate) nrows: usize,

    /// The number of columns in the tile, not exceeding `TILE_SIZE`.
    pub(crate) ncols: usize,

    /// Flag to specify if the matrix is transposed. For transposed
    /// matrices, the terms row and column need to be swapped.
    pub(crate) transposed: bool,

}


/// # TileGPU
macro_rules! impl_tile {
    ($s:ty,
     $geam:path,
     $gemm:path) => {
        impl TileGPU<$s>
        {
            /// Constructs a new `TileGPU` with the specified number of rows and
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
                let dev_ptr = DevPtr::<$s>::malloc(size).unwrap();

                let data = dev_ptr.as_slice_mut();
                for i in 0..size {
                    data[i] = init;
                }

                Self { dev_ptr, nrows, ncols, transposed: false}
            }

            pub fn data(&self) -> &mut [$s] {
                self.dev_ptr.as_slice_mut()
            }

            pub fn prefetch(&self, dev: &cuda::Device) {
                let stream = cuda::Stream::new().unwrap();
                self.dev_ptr.prefetch(self.nrows*self.ncols, &dev, &stream).unwrap()
            }


            /// Constructs a `TileGPU` from a slice of data, given the number of
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
                let dev_ptr = DevPtr::malloc(size).unwrap();
                let data = dev_ptr.as_slice_mut();
                for j in 0..ncols {
                    for i in 0..nrows {
                        data[i+j*nrows] = other[i + j*lda];
                    }
                }
                Self { dev_ptr, nrows, ncols, transposed: false }
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
                let transposed = self.transposed;
                let data = self.data();
                match transposed {
                    false => {
                        for j in 0..self.ncols {
                            let shift_tile = j*self.nrows;
                            let shift_array = j*lda;
                            other[shift_array..(self.nrows + shift_array)].copy_from_slice(&data[shift_tile..(self.nrows + shift_tile)]);
                        }},
                    true => {
                        for i in 0..self.nrows {
                            let shift_array = i*lda;
                            for j in 0..self.ncols {
                                let shift_tile = j*self.nrows;
                                other[j + shift_array] = data[i + shift_tile];
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
                let transposed = self.transposed;
                let data = self.data();
                match transposed {
                    false => unsafe { data.get_unchecked(i + j * self.nrows) },
                    true  => unsafe { data.get_unchecked(j + i * self.nrows) },
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
                let transposed = self.transposed;
                let data = self.data();
                match transposed {
                    false => unsafe { data.get_unchecked_mut(i + j * self.nrows) },
                    true  => unsafe { data.get_unchecked_mut(j + i * self.nrows) },
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
                let mut new_tile = Self::from(&self.data(),
                     self.nrows, self.ncols, self.nrows);
                new_tile.transposed = !self.transposed;
                new_tile
            }

            /// Rescale the tile
            #[inline]
            pub fn scale_mut(&mut self, cublas: &cublas::Context, factor: $s) {
                let dev_ptr_in = self.dev_ptr.clone();

                $geam(cublas, b'N', b'N', self.nrows, self.ncols,
                  factor, &dev_ptr_in, self.nrows, 0., &dev_ptr_in, self.nrows,
                  &mut self.dev_ptr, self.nrows).unwrap();

            }

            /// Returns a copy of the tile, rescaled
            #[inline]
            pub fn scale(&self, cublas: &cublas::Context, factor: $s) -> Self {

                let mut result = Self::new(self.nrows, self.ncols, 0.0);
                /*
                for x in &mut result.data() {
                    *x = *x * factor;
                }
                 */
                $geam(cublas, b'N', b'N', self.nrows, self.ncols,
                  factor, &self.dev_ptr, self.nrows, 0., &self.dev_ptr, self.nrows,
                  &mut result.dev_ptr, result.nrows).unwrap();

                result
            }

            /// Add another tile to the tile
            #[inline]
            pub fn add_mut(&mut self, cublas: &cublas::Context, other: &Self) {
                assert_eq!(self.ncols(), other.ncols());
                assert_eq!(self.nrows(), other.nrows());
                /*
                for j in 0..self.ncols() {
                  for i in 0..self.nrows() {
                    self[(i,j)] += other[(i,j)];
                  }
                }
                 */
                let transa = b'N';
                let transb = if other.transposed() != self.transposed() { b'T' } else { b'N' };
                let dev_ptr_in = self.dev_ptr.clone();

                $geam(cublas, transa, transb, self.nrows(), self.ncols(),
                  1., &dev_ptr_in, self.nrows, 1., &other.dev_ptr, other.nrows,
                  &mut self.dev_ptr, self.nrows).unwrap();

            }

            /// Adds another tile to the tile and returns a new tile with the result.
            #[inline]
            pub fn add(&self, cublas: &cublas::Context, other: &Self) -> Self {
                assert_eq!(self.ncols(), other.ncols());
                assert_eq!(self.nrows(), other.nrows());
                let mut result: Self = self.clone();
                result.add_mut(cublas, other);
                result
            }

            /// Combines two `TileGPU`s $A$ and $B$ with coefficients $\alpha$ and
            /// $\beta$, and returns a new `TileGPU` $C$:
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
            pub fn geam (cublas: &cublas::Context, alpha: $s, a: &Self, beta: $s, b: &Self) -> Self
            {
                let mut c = Self::new(a.nrows(), a.ncols(), -1.);
                Self::geam_mut(cublas, alpha, a, beta, b, &mut c);
                c
            }

            pub fn geam_mut (cublas: &cublas::Context, alpha: $s, a: &Self, beta: $s, b: &Self, c: &mut Self)
            {
                assert_eq!(a.ncols(), b.ncols());
                assert_eq!(a.ncols(), c.ncols());
                assert_eq!(a.nrows(), b.nrows());
                assert_eq!(a.nrows(), c.nrows());
                assert!(!c.transposed);

                let transa = if a.transposed { b'T' } else { b'N' };
                let transb = if b.transposed { b'T' } else { b'N' };

                $geam(cublas, transa, transb, c.nrows(), c.ncols(),
                  alpha, &a.dev_ptr, a.nrows, beta, &b.dev_ptr, b.nrows,
                  &mut c.dev_ptr, c.nrows).unwrap();
            }



            /// Performs a BLAS GEMM operation using `TilesGPU` $A$, $B$ and $C:
            /// $$C = \alpha A \dot B + \beta C$$.
            /// `TileGPU` $C$ is mutated.
            ///
            /// # Arguments
            ///
            /// * `alpha` - $\alpha$
            /// * `a` - TileGPU $A$
            /// * `b` - TileGPU $B$
            /// * `beta` - $\beta$
            /// * `c` - TileGPU $C$
            ///
            /// # Panics
            ///
            /// Panics if the tiles don't have matching sizes.
            pub fn gemm_mut (cublas: &cublas::Context, alpha: $s, a: &Self, b: &Self, beta: $s, c: &mut Self)
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

                $gemm(cublas, transa, transb, m, n, k, alpha,
                  &a.dev_ptr, lda, &b.dev_ptr, ldb, beta,
                  &mut c.dev_ptr, ldc).unwrap();

            }


            /// Generates a new `TileGPU` $C$ which is the result of a BLAS GEMM
            /// operation between two `TilesGPU` $A$ and $B$.
            /// $$C = \alpha A \dot B$$.
            ///
            /// # Arguments
            ///
            /// * `alpha` - $\alpha$
            /// * `a` - TileGPU $A$
            /// * `b` - TileGPU $B$
            ///
            /// # Panics
            ///
            /// Panics if the tiles don't have sizes that match.
            pub fn gemm (cublas: &cublas::Context, alpha: $s, a: &Self, b: &Self) -> Self
            {
                let mut c = Self::new(a.nrows(), b.ncols(), 0.);
                Self::gemm_mut(cublas, alpha, a, b, 0.0, &mut c);
                c
            }

        }

        /// Implementation of the Index trait to allow for read access to
        /// elements in the TileGPU using array indexing syntax.
        impl std::ops::Index<(usize,usize)> for TileGPU<$s>
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
                let transposed = self.transposed;
                let data = self.data();
                match transposed {
                    false => { assert!(i < self.nrows && j < self.ncols); &data[i + j * self.nrows] },
                    true  => { assert!(j < self.nrows && i < self.ncols); &data[j + i * self.nrows] },
                }
            }
        }

        /// Implementation of the IndexMut trait to allow for write access to
        /// elements in the TileGPU using array indexing syntax.
        impl std::ops::IndexMut<(usize,usize)> for TileGPU<$s>
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
                let data = self.data();
                match transposed {
                  false => { assert!(i < nrows && j < ncols);
                          &mut data[i + j * nrows] },
                  true  => { assert!(j < nrows && i < ncols);
                          &mut data[j + i * nrows] },
                }
            }
        }


        impl Clone for TileGPU<$s> {

            fn clone(&self) -> Self {
                let mut result = Self::from(self.data(), self.nrows, self.ncols, self.nrows);
                result.transposed = self.transposed;
                result
            }

        }

        impl PartialEq for TileGPU<$s> {

            fn eq(&self, other: &Self) -> bool {
                if  self.nrows == other.nrows &&
                    self.ncols == other.ncols &&
                    self.transposed == other.transposed {
                    let self_data = self.data();
                    let other_data = other.data();
                    for i in 0..(self.nrows*self.ncols) {
                        if self_data[i] != other_data[i] {
                            return false;
                        }
                    }
                    true
                } else {
                    false
                }
            }
        }

    } // rule
}


impl_tile!(f32, cublas::sgeam, cublas::sgemm);
impl_tile!(f64, cublas::dgeam, cublas::dgemm);



// ------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    // Tests for the TileGPU implementation...

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
                let tile = TileGPU::<$s>::new(10, 20, 1.0);
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
                let mut tile = TileGPU::<$s>::from(&other, 10, 15, 20);

                let mut other = vec![0.0 ; 150];
                tile.copy_in_vec(&mut other, 10);
                assert_eq!(other, other_ref);
            }

            #[test]
            #[should_panic]
            fn $creation_too_large() {
                let _ = TileGPU::<$s>::new(TILE_SIZE+1, 10, 1.0);
            }

            #[test]
            #[should_panic]
            fn $row_overflow() {
                let tile = TileGPU::<$s>::new(10, 20, 1.0);
                let _ = tile[(11,10)];
            }

            #[test]
            #[should_panic]
            fn $col_overflow() {
                let tile = TileGPU::<$s>::new(10, 20, 1.0);
                let _ = tile[(1,21)];
            }

            #[test]
            fn $index_mut() {
                let mut tile = TileGPU::<$s>::new(10, 20, 1.0);
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
                assert_eq!(tile.data(), ref_val);

            }

            #[test]
            fn $transposition() {
                let mut tile = TileGPU::<$s>::new(10, 20, 1.0);
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
                let mut tile = TileGPU::<$s>::new(10, 20, 1.0);
                for i in 0..10 {
                    for j in 0..20 {
                        tile[(i,j)] = (i as $s) * 100.0 + (j as $s);
                    }
                }
                let mut data_ref = Vec::from(tile.data());
                for i in 0..10 {
                    for j in 0..20 {
                        data_ref[i+j*10] *= 2.0;
                    }
                }
                let handle = cublas::Context::new().unwrap();
                let mut new_tile = tile.scale(&handle, 2.0);
                drop(handle);
                assert_eq!(new_tile.data(), data_ref);

                let handle = cublas::Context::new().unwrap();
                tile.scale_mut(&handle, 2.0);
                drop(handle);
                assert_eq!(tile.data(), data_ref);
            }

            #[test]
            #[should_panic]
            fn $geam_wrong_size() {
                let n = 10;
                let m = 5;
                let a = TileGPU::<$s>::new(m, n, 0.0);
                let b = TileGPU::<$s>::new(n, m, 0.0);
                let handle = cublas::Context::new().unwrap();
                let _ = TileGPU::<$s>::geam(&handle, 1.0, &a, 1.0, &b);
            }

            #[test]
            fn $geam_cases() {
                let n = 8;
                let m = 4;
                let mut a   = TileGPU::<$s>::new(m, n, 0.0);
                let mut a_t = TileGPU::<$s>::new(n, m, 0.0);
                let mut b   = TileGPU::<$s>::new(m, n, 0.0);
                let mut b_t = TileGPU::<$s>::new(n, m, 0.0);
                let zero_tile = TileGPU::<$s>::new(m, n, 0.0);
                let zero_tile_t = TileGPU::<$s>::new(n, m, 0.0);
                for i in 0..m {
                    for j in 0..n {
                        a[(i,j)] = (i * 10 + j) as $s;
                        b[(i,j)] = (i * 1000 + j*10) as $s;
                        a_t[(j,i)] = (i * 10 + j) as $s;
                        b_t[(j,i)] = (i * 1000 + j*10) as $s;
                    }
                }

                let handle = cublas::Context::new().unwrap();
                let tile = TileGPU::<$s>::geam(&handle, 0.0, &a, 0.0, &b);
                drop(handle);
                assert_eq!( tile, zero_tile);

                let handle = cublas::Context::new().unwrap();
                let tile = TileGPU::<$s>::geam(&handle, 1.0, &a, 0.0, &b);
                drop(handle);
                assert_eq!( tile, a);

                let handle = cublas::Context::new().unwrap();
                let tile = TileGPU::<$s>::geam(&handle, 0.0, &a, 1.0, &b);
                drop(handle);
                assert_eq!( tile, b);

                let mut r = TileGPU::<$s>::new(m, n, 0.0);
                for i in 0..m {
                  for j in 0..n {
                    r[(i,j)] = 2.0*a[(i,j)];
                  }
                };
                let handle = cublas::Context::new().unwrap();
                let tile = TileGPU::<$s>::geam(&handle, 2.0, &a, 0.0, &b);
                drop(handle);
                assert_eq!( tile, r );

                let handle = cublas::Context::new().unwrap();
                let tile = TileGPU::<$s>::geam(&handle, 0.5, &a, 0.5, &a);
                drop(handle);
                assert_eq!( tile, a);

                let handle = cublas::Context::new().unwrap();
                let tile = TileGPU::<$s>::geam(&handle, 0.5, &a, -0.5, &a);
                drop(handle);
                assert_eq!( tile, zero_tile);

                let handle = cublas::Context::new().unwrap();
                let tile = TileGPU::<$s>::geam(&handle, 1.0, &a, -1.0, &a);
                drop(handle);
                assert_eq!( tile, zero_tile);

                let mut r = TileGPU::<$s>::new(m, n, 0.0);
                for i in 0..m {
                  for j in 0..n {
                    r[(i,j)] = 2.0*b[(i,j)];
                  }
                };
                let handle = cublas::Context::new().unwrap();
                let tile = TileGPU::<$s>::geam(&handle, 0.0, &a, 2.0, &b);
                drop(handle);
                assert_eq!( tile, r);

                let mut r = TileGPU::<$s>::new(m, n, 0.0);
                for i in 0..m {
                  for j in 0..n {
                    r[(i,j)] = a[(i,j)] + b[(i,j)];
                  }
                };
                let handle = cublas::Context::new().unwrap();
                let tile = TileGPU::<$s>::geam(&handle, 1.0, &a, 1.0, &b);
                drop(handle);
                assert_eq!( tile, r );

                let mut r = TileGPU::<$s>::new(m, n, 0.0);
                for i in 0..m {
                    for j in 0..n {
                        r[(i,j)] = b[(i,j)] - a[(i,j)];
                    }
                };
                let handle = cublas::Context::new().unwrap();
                let tile = TileGPU::<$s>::geam(&handle, -1.0, &a, 1.0, &b);
                drop(handle);
                assert_eq!( tile, r );

                let mut r = TileGPU::<$s>::new(m, n, 0.0);
                for i in 0..m {
                    for j in 0..n {
                        r[(i,j)] = a[(i,j)] - b[(i,j)];
                    }
                };
                let handle = cublas::Context::new().unwrap();
                let tile = TileGPU::<$s>::geam(&handle, 1.0, &a, -1.0, &b);
                drop(handle);
                assert_eq!( tile, r);

                let handle = cublas::Context::new().unwrap();
                let tile = TileGPU::<$s>::geam(&handle, 1.0, &a, -1.0, &a_t.t());
                drop(handle);
                assert_eq!( tile, zero_tile);

                let handle = cublas::Context::new().unwrap();
                let tile = TileGPU::<$s>::geam(&handle, 1.0, &a_t.t(), -1.0, &a);
                drop(handle);
                assert_eq!( tile, zero_tile);

                // Mutable geam

                let handle = cublas::Context::new().unwrap();
                let mut c1 = TileGPU::<$s>::geam(&handle, 1.0, &a, 1.0, &b);
                let mut c  = TileGPU::<$s>::geam(&handle, 1.0, &a, 1.0, &b);
                TileGPU::<$s>::geam_mut(&handle, -1.0, &a, 1.0, &c1, &mut c);
                drop(handle);
                assert_eq!( c , b );

                for (alpha, beta) in [ (1.0,1.0), (1.0,-1.0), (-1.0,-1.0), (-1.0,1.0),
                                       (1.0,0.0), (0.0,-1.0), (0.5, 1.0), (0.5, 1.0),
                                       (0.5,-0.5) ] {
                        let nrows = a.nrows();
                        let ncols = a.ncols();
                        let mut c = TileGPU::<$s>::from(a.data(), nrows,
                              ncols, nrows);
                        let handle = cublas::Context::new().unwrap();
                        TileGPU::<$s>::geam_mut(&handle, alpha, &a, beta, &b, &mut c);
                        let tile = TileGPU::<$s>::geam(&handle, alpha, &a, beta, &b);
                        drop(handle);
                        assert_eq!( tile, c);
                                       }
            }

            #[test]
            fn $gemm() {
    let time = std::time::Instant::now();
                let a = TileGPU::<$s>::from(&[1. , 1.1, 1.2, 1.3,
                                    2. , 2.1, 2.2, 2.3,
                                    3. , 3.1, 3.2, 3.3],
                                    4, 3, 4).t();
    let duration = time.elapsed();
    println!("Time elapsed in a: {:?}", duration);
    let time = std::time::Instant::now();

                let b = TileGPU::<$s>::from( &[ 1.0, 2.0, 3.0, 4.0,
                                    1.1, 2.1, 3.1, 4.1],
                                    4, 2, 4 );

    let duration = time.elapsed();
    println!("Time elapsed in b: {:?}", duration);
    let time = std::time::Instant::now();

                let c_ref = TileGPU::<$s>::from( &[12.0 , 22.0 , 32.0,
                                        12.46, 22.86, 33.26],
                                        3, 2, 3);

    let duration = time.elapsed();
    println!("Time elapsed in c_ref: {:?}", duration);
    let time = std::time::Instant::now();

                let handle = cublas::Context::new().unwrap();
                let mut c = TileGPU::<$s>::gemm(&handle, 1.0, &a, &b);

    let duration = time.elapsed();
    println!("Time elapsed in gemm: {:?}", duration);
    let time = std::time::Instant::now();

                let mut difference = TileGPU::<$s>::geam(&handle, 1.0, &c, -1.0, &c_ref);
                drop(handle);

    let duration = time.elapsed();
    println!("Time elapsed in diff: {:?}", duration);
    let time = std::time::Instant::now();

                for j in 0..2 {
                    for i in 0..3 {
                        assert!(num::abs(difference[(i,j)] / c[(i,j)]) < <$s>::EPSILON);
                    }
                }

    let duration = time.elapsed();
    println!("Time elapsed in assert: {:?}", duration);
    let time = std::time::Instant::now();

                let a = a.t();
                let b = b.t();

    let duration = time.elapsed();
    println!("Time elapsed in transpositions: {:?}", duration);
    let time = std::time::Instant::now();

                let handle = cublas::Context::new().unwrap();
                let mut c_t = TileGPU::<$s>::gemm(&handle, 1.0, &b, &a);

    let duration = time.elapsed();
    println!("Time elapsed in gemm: {:?}", duration);
    let time = std::time::Instant::now();

                let mut difference = TileGPU::<$s>::geam(&handle, 1.0, &c_t, -1.0, &c.t());
                drop(handle);

    let duration = time.elapsed();
    println!("Time elapsed in gamm: {:?}", duration);
    let time = std::time::Instant::now();

                for j in 0..3 {
                    for i in 0..2 {
                        assert_eq!(difference[(i,j)], 0.0);
                    }
                }

    let duration = time.elapsed();
    println!("Time elapsed in assert: {:?}", duration);
    let time = std::time::Instant::now();

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



