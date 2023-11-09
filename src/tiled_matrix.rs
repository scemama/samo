use crate::tile;
use crate::tile::Tile;
use num::traits::Float;


/// A `TiledMatrix` is a two-dimensional data structure that divides a
/// matrix into smaller blocks or 'tiles'.  This tiling approach is
/// beneficial for algorithms that can exploit data locality, like
/// many matrix operations in numerical linear algebra, as it can
/// significantly improve cache efficiency.
///
/// The `TiledMatrix` struct is generic over `T`, which is the type of
/// the elements stored in the matrix.  It is bounded by traits that
/// ensure `T` is a `Float`.
#[derive(Debug,PartialEq,Clone)]
pub struct TiledMatrix<T>
where
    T: Float
{
    /// The total number of rows in the matrix.
    nrows: usize,

    /// The total number of columns in the matrix.
    ncols: usize,

    /// A vector of `Tile<T>` structs that collectively represent the
    /// matrix.
    tiles: Vec<Tile<T>>,

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

impl<T> TiledMatrix<T>
where
    T: Float
{
    /// Constructs a new `TiledMatrix` with the specified number of
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
    /// use samo::tiled_matrix::TiledMatrix;
    /// 
    /// let matrix = TiledMatrix::new(100, 200, 1.0);
    /// ```
    pub fn new(nrows: usize, ncols: usize, init: T) -> Self {
        const TILE_SIZE : usize = tile::TILE_SIZE;
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
        let mut tiles = Vec::<Tile<T>>::with_capacity(size);
        for _ in 0..ncols_tiles_full {
            for _ in 0..nrows_tiles_full {
                tiles.push( Tile::<T>::new(TILE_SIZE,TILE_SIZE,init) );
            }
            if nrows_tiles > nrows_tiles_full { 
                tiles.push( Tile::<T>::new(nrows_border,TILE_SIZE,init) );
            }
        }
        if ncols_tiles > ncols_tiles_full {
            for _ in 0..nrows_tiles_full {
                tiles.push( Tile::<T>::new(TILE_SIZE,ncols_border,init) );
            }
            if nrows_tiles > nrows_tiles_full { 
                tiles.push( Tile::<T>::new(nrows_border,ncols_border,init) );
            }
        }
        let transposed = false;
        TiledMatrix {
            nrows, ncols, tiles, nrows_tiles, ncols_tiles, transposed,
        }
    }


    /// Constructs a `TiledMatrix` from a slice of data, arranging the
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
    /// use samo::tiled_matrix::TiledMatrix;
    /// 
    /// let flat_matrix = vec![1.0; 100 * 200];
    /// let matrix = TiledMatrix::from(&flat_matrix, 100, 200, 100);
    /// ```
    pub fn from(other: &[T], nrows: usize, ncols: usize, lda: usize) -> Self {
        const TILE_SIZE : usize = tile::TILE_SIZE;
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
        let mut tiles = Vec::<Tile<T>>::with_capacity(size);
        for j in 0..ncols_tiles_full {
            let ncols_past = j*TILE_SIZE;
            let elts_from_prev_columns = ncols_past*lda;
            for i in 0..nrows_tiles_full {
                let elts_from_prev_rows    = i*TILE_SIZE;
                let shift = elts_from_prev_rows + elts_from_prev_columns;
                tiles.push( Tile::<T>::from(&other[shift..], TILE_SIZE, TILE_SIZE, lda) );
            }
            if nrows_tiles > nrows_tiles_full { 
                let shift = nrows_tiles_full*TILE_SIZE + elts_from_prev_columns;
                tiles.push( Tile::<T>::from(&other[shift..], nrows_border, TILE_SIZE, lda) );
            }
        }
        if ncols_tiles > ncols_tiles_full {
            let ncols_past = ncols_tiles_full*TILE_SIZE;
            let elts_from_prev_columns = ncols_past*lda;
            for i in 0..nrows_tiles_full {
                let elts_from_prev_rows = i*TILE_SIZE;
                let shift = elts_from_prev_rows + elts_from_prev_columns;
                tiles.push( Tile::<T>::from(&other[shift..], TILE_SIZE, ncols_border, lda) );
            }
            if nrows_tiles > nrows_tiles_full { 
                let elts_from_prev_rows = nrows_tiles_full*TILE_SIZE;
                let shift = elts_from_prev_rows + elts_from_prev_columns;
                tiles.push( Tile::<T>::from(&other[shift..], nrows_border, ncols_border, lda) );
            }
        }
        let transposed = false;
        TiledMatrix {
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
    /// use samo::tiled_matrix::TiledMatrix;
    /// 
    /// let mut flat_matrix = vec![0.0; 100 * 200];
    /// let tiled_matrix = TiledMatrix::new(100, 200, 1.0);
    /// tiled_matrix.copy_in_vec(&mut flat_matrix, 100);
    /// ```
    pub fn copy_in_vec(&self, other: &mut [T], lda:usize) {
        const TILE_SIZE : usize = tile::TILE_SIZE;

        match self.transposed {
            false => {
                for j in 0..self.ncols_tiles {
                    let elts_from_prev_columns = j*TILE_SIZE*lda;
                    for i in 0..self.nrows_tiles {
                        let elts_from_prev_rows = i*TILE_SIZE;
                        let shift = elts_from_prev_rows + elts_from_prev_columns;
                        self.tiles[i + j*self.nrows_tiles].copy_in_vec(&mut other[shift..], lda)
                    }
                }
            },
            // TODO
            true => {
                for j in 0..self.ncols_tiles {
                    let elts_from_prev_columns = j*TILE_SIZE*lda;
                    for i in 0..self.nrows_tiles {
                        let elts_from_prev_rows = i*TILE_SIZE;
                        let shift = elts_from_prev_rows + elts_from_prev_columns;
                        self.tiles[i + j*self.nrows_tiles].copy_in_vec(&mut other[shift..], lda)
                    }
                }
            },
        }
    }


    /// Returns the number of rows in the matrix.
    pub fn nrows(&self) -> usize {
        match self.transposed {
            false => self.nrows, 
            true  => self.ncols,
        }
    }

    /// Returns the number of columns in the matrix.
    pub fn ncols(&self) -> usize {
        match self.transposed {
            false => self.ncols, 
            true  => self.nrows,
        }
    }

    /// Returns the number of rows of tiles in the matrix.
    pub fn nrows_tiles(&self) -> usize {
        match self.transposed {
            false => self.nrows_tiles, 
            true  => self.ncols_tiles,
        }
    }

    /// Returns the number of columns of tiles in the matrix.
    pub fn ncols_tiles(&self) -> usize {
        match self.transposed {
            false => self.ncols_tiles, 
            true  => self.nrows_tiles,
        }
    }

    /// Tells if the matrix is transposed or not.
    pub fn transposed(&self) -> bool {
        self.transposed
    }

    /// Transposes the current matrix
    pub fn transpose_mut(&mut self) {
        self.transposed = ! self.transposed;
        for t in &mut self.tiles {
            t.transpose_mut();
        }
    }

    /// Returns the transposed of the matrix
    pub fn t(&self) -> Self {
        let mut new_tiles = self.tiles.clone();
        for t in &mut new_tiles { t.transpose_mut() };
        TiledMatrix {
            transposed: !self.transposed,
            tiles: new_tiles,
            ..(*self)
        }
    }
    
    /// Returns a reference to the tile at $(i,j)$ in the
    /// 2D-array of tiles.
    pub fn get_tile(&self, i: usize, j: usize) -> &Tile<T> {
        assert!(i < self.nrows() && j < self.ncols());
        match self.transposed {
            false => { &self.tiles[i + j*self.nrows_tiles] },
            true  => { &self.tiles[j + i*self.ncols_tiles] },
        }
    }

    /// Returns a mutable reference to the tile at $(i,j)$ in the
    /// 2D-array of tiles.
    pub fn get_tile_mut(&mut self, i: usize, j: usize) -> &mut Tile<T> {
        assert!(i < self.nrows() && j < self.ncols());
        match self.transposed {
            false => { &mut self.tiles[i + j*self.nrows_tiles] },
            true  => { &mut self.tiles[j + i*self.ncols_tiles] },
        }
    }
}

impl<T> std::ops::Index<(usize,usize)> for TiledMatrix<T>
where
    T: Float
{
    type Output = T;
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
    /// use samo::tiled_matrix::TiledMatrix;
    ///
    /// let matrix = TiledMatrix::new(64, 64, 1.0);
    /// assert_eq!(matrix[(0, 0)], 1.0);
    /// ```
    fn index(&self, (i,j): (usize,usize)) -> &Self::Output {
        assert!(i < self.nrows() && j < self.ncols());
        const TILE_SIZE : usize = tile::TILE_SIZE;
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
        &tile[(row_in_tile, col_in_tile)]
    }
}

impl<T> std::ops::IndexMut<(usize,usize)> for TiledMatrix<T>
where
    T: Float
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
    /// use samo::tiled_matrix::TiledMatrix;
    /// 
    /// let mut matrix = TiledMatrix::new(64, 64, 1.0);
    /// matrix[(0, 0)] = 2.0;
    /// assert_eq!(matrix[(0, 0)], 2.0);
    /// ```
    fn index_mut(&mut self, (i,j): (usize,usize)) -> &mut Self::Output {
        assert!(i < self.nrows() && j < self.ncols());
        const TILE_SIZE : usize = tile::TILE_SIZE;
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
        &mut tile[(row_in_tile, col_in_tile)]
    }
}



/// Performs a BLAS DGEMM operation using `TiledMatrices` $A$, $B$ and $C:
/// $$C = \alpha A \dot B + \beta C$$.
/// `TiledMatrix` $C$ is mutated.
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
pub fn dgemm_mut (alpha: f64, a: &TiledMatrix<f64>, b: &TiledMatrix<f64>, beta: f64, c: &mut TiledMatrix<f64>) {
    assert!(a.ncols() == b.nrows());
    assert!(a.nrows() == c.nrows());
    assert!(b.ncols() == c.ncols());

    for j in 0..(c.ncols_tiles()) {
        for i in 0..(c.nrows_tiles()) {
            let c_tile_mut = c.get_tile_mut(i,j);
            c_tile_mut.scale_mut(beta);
        }
        for k in 0..(a.ncols_tiles()) {
            let b_tile = b.get_tile(k,j);
            for i in 0..(c.nrows_tiles()) {
                let c_tile_mut = c.get_tile_mut(i,j);
                let a_tile = a.get_tile(i,k);
                tile::dgemm_mut(alpha, a_tile, b_tile, 1.0, c_tile_mut);
            }
        }
    }

}

/// Performs a BLAS SGEMM operation using `TiledMatrices` $A$, $B$ and $C:
/// $$C = \alpha A \dot B + \beta C$$.
/// `TiledMatrix` $C$ is mutated.
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
pub fn sgemm_mut (alpha: f32, a: &TiledMatrix<f32>, b: &TiledMatrix<f32>, beta: f32, c: &mut TiledMatrix<f32>) {
    assert!(a.ncols() == b.nrows());
    assert!(a.nrows() == c.nrows());
    assert!(b.ncols() == c.ncols());

    for j in 0..(c.ncols_tiles()) {
        for i in 0..(c.nrows_tiles()) {
            let c_tile_mut = c.get_tile_mut(i,j);
            c_tile_mut.scale_mut(beta);
            for k in 0..(a.ncols_tiles()) {
                let a_tile = a.get_tile(i,k);
                let b_tile = b.get_tile(k,j);
                tile::sgemm_mut(alpha, a_tile, b_tile, 1.0, c_tile_mut);
            }
        }
    }

}


/// Generates a new `TiledMatrix` $C$ which is the result of a BLAS DGEMM
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
pub fn dgemm (alpha: f64, a: &TiledMatrix<f64>, b: &TiledMatrix<f64>) -> TiledMatrix<f64>
{
    let mut c = TiledMatrix::new(a.nrows(), b.ncols(), 0.0f64);
    dgemm_mut(alpha, a, b, 0.0f64, &mut c);
    c
}


/// Generates a new `TiledMatrix` $C$ which is the result of a BLAS SGEMM
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
/// Panics if the tiles don't have sizes that match.
pub fn sgemm (alpha: f32, a: &TiledMatrix<f32>, b: &TiledMatrix<f32>) -> TiledMatrix<f32>
{
    let mut c = TiledMatrix::new(a.nrows(), b.ncols(), 0.0f32);
    sgemm_mut(alpha, a, b, 0.0f32, &mut c);
    c
}




//--------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;


    const TILE_SIZE : usize = tile::TILE_SIZE;

    #[test]
    fn creation() {
        let matrix = TiledMatrix::new(1000, 2000, 1.0);
        assert_eq!(matrix.nrows, 1000);
        assert_eq!(matrix.ncols, 2000);
        for j in 0..(matrix.ncols) {
            for i in 0..(matrix.nrows) {
                assert_eq!(matrix[(i,j)], 1.0);
            }
        }

        let mut other_ref = vec![ 0. ; 65*166 ];
        let mut other = vec![ 0. ; 67*166 ];
        for j in 0..166 {
            for i in 0..65 {
                other_ref[i + j*65] = (i as f64) + (j as f64)*1000.0;
                other[i + j*67] = (i as f64) + (j as f64)*1000.0;
            }
        }
        let matrix = TiledMatrix::from(&other, 65, 166, 67);
        let mut converted = vec![0.0 ; 65*166];
        matrix.copy_in_vec(&mut converted, 65);
        assert_eq!(converted, other_ref);

        for j in 0..166 {
            for i in 0..65 {
                assert_eq!(other[i + j*67], matrix[(i,j)]);
            }
        }
    }

    #[test]
    fn number_of_tiles() {
        let matrix = TiledMatrix::new(2*TILE_SIZE, 10, 0.);
        assert_eq!(matrix.nrows_tiles, 2);
        let matrix = TiledMatrix::new(2*TILE_SIZE+1, 10, 0.);
        assert_eq!(matrix.nrows_tiles, 3);
        let matrix = TiledMatrix::new(10, 2*TILE_SIZE, 0.);
        assert_eq!(matrix.ncols_tiles, 2);
        let matrix = TiledMatrix::new(10, 2*TILE_SIZE+1, 0.);
        assert_eq!(matrix.ncols_tiles, 3);
    }

    #[test]
    fn transposition() {
        let m = 66;
        let n = 166;
        let mut a = vec![ 0. ; m*n ];
        let mut a_t = vec![ 0. ; m*n ];
        for j in 0..n {
            for i in 0..m {
                a  [i + j*m] = (i as f64) + (j as f64)*1000.0;
                a_t[j + i*n] = (i as f64) + (j as f64)*1000.0;
            }
        }
        let a = TiledMatrix::from(&a, m, n, m);
        let a_t = TiledMatrix::from(&a_t, n, m, n);

        let b = a_t.t();
        assert!(!a.transposed());
        assert!(!a_t.transposed());
        assert!(b.transposed());
        for j in 0..n {
            for i in 0..m {
                assert_eq!(a[(i,j)], b[(i,j)]);
            }
        }
    }

    use crate::helper_blas::{blas_dgemm};
    #[test]
    fn test_dgemm() {
        let m = 2*TILE_SIZE+1;
        let n = 3*TILE_SIZE+2;
        let k = 4*TILE_SIZE+3;

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
        blas_dgemm(b'N', b'N', m, n, k, 2.0, &a, m, &b, k, 0.0f64, &mut c_ref, m);
        let c_ref = TiledMatrix::from(&c_ref, m, n, m);

        let a = TiledMatrix::from(&a, m, k, m);
        let b = TiledMatrix::from(&b, k, n, k);
        let c = dgemm(2.0, &a, &b);
        assert_eq!(c, c_ref);
    }

}
