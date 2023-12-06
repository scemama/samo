use crate::cublas;
use crate::cuda;
use rayon::prelude::*;

use crate::tile_gpu::TILE_SIZE;
use crate::tile_gpu::TileGPU;
use cuda::Device;

/// A `TiledMatrix` is a two-dimensional data structure that divides a
/// matrix into smaller blocks or 'tiles'.  This tiling approach is
/// beneficial for algorithms that can exploit data locality, like
/// many matrix operations in numerical linear algebra, as it can
/// significantly improve cache efficiency.
///
/// The `TiledMatrix` struct is generic over `T`, which is the type of
/// the elements stored in the matrix.
/// ensure `T` is a `Float`.
#[derive(Debug)]
pub struct TiledMatrixGPU<T: Clone + PartialEq>
{
    /// The total number of rows in the matrix.
    nrows: usize,

    /// The total number of columns in the matrix.
    ncols: usize,

    /// A vector of `TileGPU<T>` that collectively represent the
    /// matrix.
    tiles: Vec<TileGPU<T>>,

    /// The number of rows of tiles, not individual elements, in the
    /// matrix.
    nrows_tiles: usize,

    /// The number of columns of tiles, not individual elements, in
    /// the matrix.
    ncols_tiles: usize,


    /// Flag to specify if the matrix is transposed. For transposed
    /// matrices, the terms row and column need to be swapped.
    transposed: bool,
}

macro_rules! impl_tiled_matrix {
    ($s:ty) => {
        impl TiledMatrixGPU<$s>
        {
            /// Constructs a new `TiledMatrixGPU` with the specified number of
            /// rows and columns, initializing all tiles with the provided
            /// `init` value.
            ///
            /// # Arguments
            ///
            /// * `nrows` - The total number of rows in the matrix.
            /// * `ncols` - The total number of columns in the matrix.
            /// * `init` - Initial value to set all elements of the matrix.
            ///
            /// # Examples
            ///
            /// ```
            /// use samo::tiled_matrix_gpu::TiledMatrixGPU;
            ///
            /// let matrix = TiledMatrixGPU::<f64>::new(100, 200, 1.0);
            /// ```
            pub fn new(nrows: usize, ncols: usize, init: $s) -> Self {
                let nrows_tiles_full = nrows / TILE_SIZE;
                let ncols_tiles_full = ncols / TILE_SIZE;

                let nrows_tiles =
                    if nrows_tiles_full * TILE_SIZE < nrows {
                        nrows_tiles_full + 1
                    } else {
                        nrows_tiles_full
                    };

                let ncols_tiles =
                    if ncols_tiles_full * TILE_SIZE < ncols {
                        ncols_tiles_full + 1
                    } else {
                        ncols_tiles_full
                    };

                let nrows_border = nrows - nrows_tiles_full * TILE_SIZE;
                let ncols_border = ncols - ncols_tiles_full * TILE_SIZE;
                let size = nrows_tiles * ncols_tiles;
                let mut tiles = Vec::<_>::with_capacity(size);
                for _ in 0..ncols_tiles_full {
                    for _ in 0..nrows_tiles_full {
                        tiles.push( (TILE_SIZE,TILE_SIZE,init) );
                    }
                    if nrows_tiles > nrows_tiles_full {
                        tiles.push( (nrows_border,TILE_SIZE,init) );
                    }
                }
                if ncols_tiles > ncols_tiles_full {
                    for _ in 0..nrows_tiles_full {
                        tiles.push( (TILE_SIZE,ncols_border,init) );
                    }
                    if nrows_tiles > nrows_tiles_full {
                        tiles.push( (nrows_border,ncols_border,init) );
                    }
                }

                // Parallel generation of tiles
                let tiles = tiles.par_iter().map(
                    |(nrows, ncols, init)| { TileGPU::<$s>::new(*nrows, *ncols, *init) }
                ).collect();

                let transposed = false;
                TiledMatrixGPU {
                    nrows, ncols, tiles, nrows_tiles, ncols_tiles, transposed,
                }
            }


            /// Constructs a `TiledMatrixGPU` from a slice of data, arranging the
            /// data into tiles based on the leading dimension (`lda`) of the
            /// input.  This method is particularly useful for converting a
            /// flattened data array into a tiled matrix format.
            ///
            /// # Arguments
            ///
            /// * `other` - The slice from which to construct the tiled matrix.
            /// * `nrows` - The total number of rows in the matrix.
            /// * `ncols` - The total number of columns in the matrix.
            /// * `lda` - The leading dimension of the input data in `other`.
            ///
            /// # Examples
            ///
            /// ```
            /// use samo::tiled_matrix_gpu::TiledMatrixGPU;
            ///
            /// let flat_matrix = vec![1.0; 100 * 200];
            /// let matrix = TiledMatrixGPU::<f64>::from(&flat_matrix, 100, 200, 100);
            /// ```
            pub fn from(other: &[$s], nrows: usize, ncols: usize, lda: usize) -> Self {
                let nrows_tiles_full = nrows / TILE_SIZE;
                let ncols_tiles_full = ncols / TILE_SIZE;

                let nrows_tiles =
                    if nrows_tiles_full * TILE_SIZE < nrows {
                        nrows_tiles_full + 1
                    } else {
                        nrows_tiles_full
                    };

                let ncols_tiles =
                    if ncols_tiles_full * TILE_SIZE < ncols {
                        ncols_tiles_full + 1
                    } else {
                        ncols_tiles_full
                    };

                let nrows_border = nrows - nrows_tiles_full * TILE_SIZE;
                let ncols_border = ncols - ncols_tiles_full * TILE_SIZE;
                let size = nrows_tiles * ncols_tiles;

                let mut tiles = Vec::<_>::with_capacity(size);
                for j in 0..ncols_tiles_full {
                    let ncols_past = j*TILE_SIZE;
                    let elts_from_prev_columns = ncols_past*lda;
                    for i in 0..nrows_tiles_full {
                        let elts_from_prev_rows    = i*TILE_SIZE;
                        let shift = elts_from_prev_rows + elts_from_prev_columns;
                        tiles.push((shift, TILE_SIZE, TILE_SIZE, lda));
                    }
                    if nrows_tiles > nrows_tiles_full {
                        let shift = nrows_tiles_full*TILE_SIZE + elts_from_prev_columns;
                        tiles.push((shift, nrows_border, TILE_SIZE, lda));
                    }
                }
                if ncols_tiles > ncols_tiles_full {
                    let ncols_past = ncols_tiles_full*TILE_SIZE;
                    let elts_from_prev_columns = ncols_past*lda;
                    for i in 0..nrows_tiles_full {
                        let elts_from_prev_rows = i*TILE_SIZE;
                        let shift = elts_from_prev_rows + elts_from_prev_columns;
                        tiles.push((shift, TILE_SIZE, ncols_border, lda));
                    }
                    if nrows_tiles > nrows_tiles_full {
                        let elts_from_prev_rows = nrows_tiles_full*TILE_SIZE;
                        let shift = elts_from_prev_rows + elts_from_prev_columns;
                        tiles.push((shift, nrows_border, ncols_border, lda));
                    }
                }

                // Parallel generation of tiles
                let tiles = tiles.par_iter().map(
                |(shift, nrows, ncols, lda)| { TileGPU::<$s>::from(&other[*shift..], *nrows, *ncols, *lda) }
                ).collect();

                let transposed = false;
                TiledMatrixGPU {
                    nrows, ncols, tiles, nrows_tiles, ncols_tiles, transposed,
                }
            }

            /// Copies the elements of the tiled matrix into a provided
            /// mutable slice, preserving the original two-dimensional layout.
            /// This method can be used to convert the tiled matrix back into
            /// a flat array format.
            ///
            /// # Arguments
            ///
            /// * `other` - The mutable slice into which the tiled matrix's data will be copied.
            /// * `lda` - The leading dimension to be used when copying the data into `other`.
            ///
            /// # Panics
            ///
            /// Panics if `other` is not large enough to hold the entire matrix.
            ///
            /// # Examples
            ///
            /// ```
            /// use samo::tiled_matrix_gpu::TiledMatrixGPU;
            ///
            /// let mut flat_matrix = vec![0.0; 100 * 200];
            /// let tiled_matrix = TiledMatrixGPU::<f64>::new(100, 200, 1.0);
            /// tiled_matrix.copy_in_vec(&mut flat_matrix, 100);
            /// ```
            pub fn copy_in_vec(&self, other: &mut [$s], lda:usize) {
                self.prefetch(&Device::CPU);
                match self.transposed {
                    false => {
                        other.par_chunks_mut(lda).enumerate().for_each(|(j,col)| {
                            let col_tile = j / TILE_SIZE;
                            let col_in_tile = j - col_tile * TILE_SIZE;
                            for row_tile in 0..self.nrows_tiles {
                                let tile = &self.tiles[row_tile+col_tile*self.nrows_tiles];
                                let i0 = row_tile*TILE_SIZE;
                                let s = col_in_tile * tile.nrows;
                                let data = tile.data();
                                for i in 0..tile.nrows {
                                    col[i0+i] = data[s+i];
                                }
                            }
                        });
                    },
                    true  => {
                        other.par_chunks_mut(lda).enumerate().for_each(|(j,col)| {
                            let row_tile = j / TILE_SIZE;
                            let col_in_tile = j - row_tile * TILE_SIZE;
                            for col_tile in 0..(self.ncols_tiles) {
                                let tile = &self.tiles[row_tile + col_tile*self.nrows_tiles];
                                let i0 = col_tile*TILE_SIZE;
                                let s = col_in_tile ;
                                let data = tile.data();
                                for i in 0..tile.ncols {
                                    col[i0+i] = data[s+i*tile.nrows];
                                }
                            }
                        });
                    }
                };

            }


            /// Returns the number of rows in the matrix.
            #[inline]
            pub fn nrows(&self) -> usize {
                match self.transposed {
                    false => self.nrows,
                    true  => self.ncols,
                }
            }

            /// Returns the number of columns in the matrix.
            #[inline]
            pub fn ncols(&self) -> usize {
                match self.transposed {
                    false => self.ncols,
                    true  => self.nrows,
                }
            }

            /// Returns the number of rows of tiles in the matrix.
            #[inline]
            pub fn nrows_tiles(&self) -> usize {
                match self.transposed {
                    false => self.nrows_tiles,
                    true  => self.ncols_tiles,
                }
            }

            /// Returns the number of columns of tiles in the matrix.
            #[inline]
            pub fn ncols_tiles(&self) -> usize {
                match self.transposed {
                    false => self.ncols_tiles,
                    true  => self.nrows_tiles,
                }
            }

            /// Tells if the matrix is transposed or not.
            #[inline]
            pub fn transposed(&self) -> bool {
                self.transposed
            }

            /// Transposes the current matrix
            #[inline]
            pub fn transpose_mut(&mut self) {
                self.transposed = ! self.transposed;
                for t in &mut self.tiles {
                    t.transpose_mut();
                }
            }

            /// Returns the transposed of the matrix
            #[inline]
            pub fn t(&self) -> Self {
                let mut new_tiles = self.tiles.clone();
                for t in &mut new_tiles { t.transpose_mut() };
                TiledMatrixGPU {
                    transposed: !self.transposed,
                    tiles: new_tiles,
                    ..(*self)
                }
            }

            /// Returns a reference to the tile at $(i,j)$ in the
            /// 2D-array of tiles.
            #[inline]
            pub fn get_tile(&self, i: usize, j: usize) -> &TileGPU<$s> {
                assert!(i < self.nrows() && j < self.ncols());
                match self.transposed {
                    false => { &self.tiles[i + j*self.nrows_tiles] },
                    true  => { &self.tiles[j + i*self.nrows_tiles] },
                }
            }

            /// Returns a mutable reference to the tile at $(i,j)$ in the
            /// 2D-array of tiles.
            #[inline]
            pub fn get_tile_mut(&mut self, i: usize, j: usize) -> &mut TileGPU<$s> {
                assert!(i < self.nrows() && j < self.ncols());
                match self.transposed {
                    false => { &mut self.tiles[i + j*self.nrows_tiles] },
                    true  => { &mut self.tiles[j + i*self.nrows_tiles] },
                }
            }

            pub fn prefetch(&self, dev: &Device) {
                for tile in self.tiles.iter() {
                  tile.prefetch(dev);
                }
            }
            /// Generates a new `TiledMatrixGPU` $C$ which is the result of a BLAS DGEMM
            /// operation between two `TiledMatrices` $A$ and $B$.
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
            /// Panics if the `TiledMatrices` don't have matching sizes.
            pub fn gemm (alpha: $s, a: &Self, b: &Self) -> Self
            {
                let mut c = Self::new(a.nrows(), b.ncols(), 0.0);
                Self::gemm_mut(alpha, a, b, 0.0, &mut c);
                c
            }


            /// Performs a BLAS GEMM operation using `TiledMatrices` $A$, $B$ and $C:
            /// $$C = \alpha A \dot B + \beta C$$.
            /// `TiledMatrixGPU` $C$ is mutated.
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
            pub fn gemm_mut(alpha: $s, a: &Self, b: &Self, beta: $s, c: &mut Self)
            {
                assert!(!c.transposed);
                assert_eq!(a.ncols(), b.nrows());
                assert_eq!(a.nrows(), c.nrows());
                assert_eq!(b.ncols(), c.ncols());

                assert_eq!(a.ncols_tiles(), b.nrows_tiles());
                assert_eq!(a.nrows_tiles(), c.nrows_tiles());
                assert_eq!(b.ncols_tiles(), c.ncols_tiles());

                let nrows_tiles: usize = c.nrows_tiles();
                let ncols_tiles: usize = c.ncols_tiles();

                let dev = cuda::get_device().unwrap();

                c.tiles.par_chunks_mut(nrows_tiles).enumerate().for_each(|(j,row)| {

                    let cublas = cublas::Context::new().unwrap();
                    let s = vec![ cuda::Stream::new().unwrap() ; a.ncols_tiles() ];

                    for k in 0..(b.nrows_tiles) {
                        b.get_tile(k,j).prefetch(&dev);
                    }

                    row.iter_mut().enumerate().for_each(|(i,cij)| {
                        if (i < nrows_tiles - 1) { a.get_tile(i+1,0).prefetch(&dev); }

                        s[0].set_active(&cublas).unwrap();
                        let b_tile = b.get_tile(0,j);
                        let a_tile = a.get_tile(i,0);
                        TileGPU::<$s>::gemm_mut(&cublas, alpha, a_tile, b_tile, beta, cij);

                        for k in 1..(a.ncols_tiles()) {
                            s[k].set_active(&cublas).unwrap();
                            let b_tile = b.get_tile(k,j);
                            let a_tile = a.get_tile(i,k);
                            if (i < nrows_tiles - 1) { a.get_tile(i+1,k).prefetch(&dev); }
                            TileGPU::<$s>::gemm_mut(&cublas, alpha, a_tile, b_tile, 1.0, cij);
                        }
                    });

                });
            }
        }

        impl PartialEq for TiledMatrixGPU<$s> {
            fn eq(&self, other: &Self) -> bool {
                if  self.nrows == other.nrows &&
                    self.ncols == other.ncols &&
                    self.nrows_tiles == other.nrows_tiles &&
                    self.ncols_tiles == other.ncols_tiles &&
                    self.transposed == other.transposed {
                    for i in 0..(self.nrows_tiles*self.ncols_tiles) {
                        if self.tiles[i] != other.tiles[i] {
                            return false;
                        }
                    }
                    true
                } else {
                    false
                }
            }
        }

        impl std::ops::Index<(usize,usize)> for TiledMatrixGPU<$s>
        {
            type Output = $s;
            /// Provides immutable access to the element at the specified
            /// (row, column) index.  This method calculates the corresponding
            /// tile and the index within that tile to return a reference to
            /// the element.
            ///
            /// # Arguments
            ///
            /// * `(i, j)` - A tuple containing the row and column indices for the element.
            ///
            /// # Panics
            ///
            /// Panics if the specified indices are out of bounds of the matrix dimensions.
            ///
            /// # Examples
            ///
            /// ```
            /// use samo::tiled_matrix_gpu::TiledMatrixGPU;
            ///
            /// let matrix = TiledMatrixGPU::<f64>::new(64, 64, 1.0);
            /// assert_eq!(matrix[(0, 0)], 1.0);
            /// ```
            #[inline]
            fn index(&self, (i,j): (usize,usize)) -> &Self::Output {
                assert!(i < self.nrows() && j < self.ncols());
                let row_tile = i / TILE_SIZE;
                let col_tile = j / TILE_SIZE;
                let row_in_tile = i - row_tile*TILE_SIZE;
                let col_in_tile = j - col_tile*TILE_SIZE;

                let tile = match self.transposed {
                    false => {
                        assert!(row_tile < self.nrows_tiles && col_tile < self.ncols_tiles);
                        &self.tiles[ row_tile + col_tile * self.nrows_tiles ] },
                    true  => {
                        assert!(col_tile < self.nrows_tiles && row_tile < self.ncols_tiles);
                        &self.tiles[ col_tile + row_tile * self.nrows_tiles ] },
                };
                tile.prefetch(&cuda::Device::CPU);
                &tile[(row_in_tile, col_in_tile)]
            }
        }

        impl std::ops::IndexMut<(usize,usize)> for TiledMatrixGPU<$s>
        {
            /// Provides mutable access to the element at the specified (row, column) index.
            /// This method calculates the corresponding tile and the index within that tile to return a mutable reference to the element.
            ///
            /// # Arguments
            ///
            /// * `(i, j)` - A tuple containing the row and column indices for the element.
            ///
            /// # Panics
            ///
            /// Panics if the specified indices are out of bounds of the matrix dimensions.
            ///
            /// # Examples
            ///
            /// ```
            /// use samo::tiled_matrix_gpu::TiledMatrixGPU;
            ///
            /// let mut matrix = TiledMatrixGPU::<f64>::new(64, 64, 1.0);
            /// matrix[(0, 0)] = 2.0;
            /// assert_eq!(matrix[(0, 0)], 2.0);
            /// ```
            #[inline]
            fn index_mut(&mut self, (i,j): (usize,usize)) -> &mut Self::Output {
                assert!(i < self.nrows() && j < self.ncols());
                let row_tile = i / TILE_SIZE;
                let col_tile = j / TILE_SIZE;
                let row_in_tile = i - row_tile*TILE_SIZE;
                let col_in_tile = j - col_tile*TILE_SIZE;
                let tile = match self.transposed {
                    false => {
                        assert!(row_tile < self.nrows_tiles && col_tile < self.ncols_tiles);
                        &mut self.tiles[ row_tile + col_tile * self.nrows_tiles ] },
                    true  => {
                        assert!(col_tile < self.nrows_tiles && row_tile < self.ncols_tiles);
                        &mut self.tiles[ col_tile + row_tile * self.nrows_tiles ] },
                };
                tile.prefetch(&cuda::Device::CPU);
                &mut tile[(row_in_tile, col_in_tile)]
            }
        }

    }
}

impl_tiled_matrix!(f32);
impl_tiled_matrix!(f64);


//--------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use crate::blas_utils;

    #[test]
    fn creation() {
        let m = 2*TILE_SIZE+1;
        let n = 2*TILE_SIZE+2;
        let matrix = TiledMatrixGPU::<f64>::new(m, n, 1.0);
        assert_eq!(matrix.nrows, m);
        assert_eq!(matrix.ncols, n);
        for j in 0..(matrix.ncols) {
            for i in 0..(matrix.nrows) {
                assert_eq!(matrix[(i,j)], 1.0);
            }
        }
    }

    #[test]
    fn copy_in_vec() {
        let m = 2*TILE_SIZE+1;
        let n = 2*TILE_SIZE+2;
        let lda = 2*TILE_SIZE+4;
        let mut other_ref = vec![ 0. ; m*n ];
        let mut other_ref_t = vec![ 0. ; n*m ];
        let mut other = vec![ 0. ; lda*n ];
        for j in 0..n {
            for i in 0..m {
                other_ref_t[j + i*n] = (i as f64) + (j as f64)*1000.0;
                other_ref[i + j*m] = (i as f64) + (j as f64)*1000.0;
                other[i + j*lda] = (i as f64) + (j as f64)*1000.0;
            }
        }
        let matrix = TiledMatrixGPU::<f64>::from(&other, m, n, lda);
        let mut converted = vec![0.0 ; m*n];
        matrix.copy_in_vec(&mut converted, m);
        assert_eq!(converted, other_ref);

        for j in 0..n {
            for i in 0..m {
                assert_eq!(other[i + j*lda], matrix[(i,j)]);
            }
        }

        let matrix = matrix.t();
        matrix.copy_in_vec(&mut converted, n);
        assert_eq!(converted, other_ref_t);

    }

    #[test]
    fn number_of_tiles() {
        let matrix = TiledMatrixGPU::<f64>::new(2*TILE_SIZE, 10, 0.);
        assert_eq!(matrix.nrows_tiles, 2);
        let matrix = TiledMatrixGPU::<f64>::new(2*TILE_SIZE+1, 10, 0.);
        assert_eq!(matrix.nrows_tiles, 3);
        let matrix = TiledMatrixGPU::<f64>::new(10, 2*TILE_SIZE, 0.);
        assert_eq!(matrix.ncols_tiles, 2);
        let matrix = TiledMatrixGPU::<f64>::new(10, 2*TILE_SIZE+1, 0.);
        assert_eq!(matrix.ncols_tiles, 3);
    }

    #[test]
    fn transposition() {
        let m = 2*TILE_SIZE+1;
        let n = 2*TILE_SIZE+2;
        let mut a = vec![ 0. ; m*n ];
        let mut a_t = vec![ 0. ; m*n ];
        for j in 0..n {
            for i in 0..m {
                a  [i + j*m] = (i as f64) + (j as f64)*1000.0;
                a_t[j + i*n] = (i as f64) + (j as f64)*1000.0;
            }
        }
        let a   = TiledMatrixGPU::<f64>::from(&a, m, n, m);
        a.prefetch(&Device::CPU);

        let a_t = TiledMatrixGPU::<f64>::from(&a_t, n, m, n);
        let b = a_t.t();
        b.prefetch(&Device::CPU);

        assert!(!a.transposed());
        assert!(!a_t.transposed());
        assert!(b.transposed());

        for j in 0..n {
            for i in 0..m {
                assert_eq!(a[(i,j)], b[(i,j)]);
            }
        }
    }

    #[test]
    #[ignore]
    fn test_dgemm() {
        let m = 2*TILE_SIZE+1;
        let n = 2*TILE_SIZE+2;
        let k = 2*TILE_SIZE+3;

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
        let mut c_ref_t = vec![ 1. ; m*n ];
        blas_utils::dgemm(b'N', b'N', m, n, k, 2.0, &a, m, &b, k, 0.0f64, &mut c_ref, m);
        blas_utils::dgemm(b'T', b'T', n, m, k, 2.0, &b, k, &a, m, 0.0f64, &mut c_ref_t, n);

        // Tiled matrices
        let c_ref = TiledMatrixGPU::<f64>::from(&c_ref, m, n, m);
        let c_ref_t = TiledMatrixGPU::<f64>::from(&c_ref_t, n, m, n);
        c_ref_t.prefetch(&Device::CPU);

        let a = TiledMatrixGPU::<f64>::from(&a, m, k, m);
        let b = TiledMatrixGPU::<f64>::from(&b, k, n, k);
        let c = TiledMatrixGPU::<f64>::gemm(2.0, &a, &b);
        c.prefetch(&Device::CPU);
        assert_eq!(c, c_ref);

        let a = a.t();
        let b = b.t();
        let c_t = TiledMatrixGPU::<f64>::gemm(2.0, &b, &a);
        c.prefetch(&Device::CPU);
        assert_eq!(c_t, c_ref_t);
    }

}
