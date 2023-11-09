use crate::tile;
use crate::tile::Tile;

/// A `TiledMatrix` is a two-dimensional data structure that divides a
/// matrix into smaller blocks or 'tiles'.  This tiling approach is
/// beneficial for algorithms that can exploit data locality, like
/// many matrix operations in numerical linear algebra, as it can
/// significantly improve cache efficiency.
///
/// The `TiledMatrix` struct is generic over `T`, which is the type of
/// the elements stored in the matrix.  It is bounded by traits that
/// ensure `T` can be copied and has a default value.
#[derive(Debug)]
pub struct TiledMatrix<T>
where
    T: Copy + Default
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
}

impl<T> TiledMatrix<T>
where
    T: Copy + Default
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
        TiledMatrix {
            nrows, ncols, tiles, nrows_tiles, ncols_tiles
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
        TiledMatrix {
            nrows, ncols, tiles, nrows_tiles, ncols_tiles
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

        for j in 0..self.ncols_tiles {
            let elts_from_prev_columns = j*TILE_SIZE*lda;
            for i in 0..self.nrows_tiles {
                let elts_from_prev_rows = i*TILE_SIZE;
                let shift = elts_from_prev_rows + elts_from_prev_columns;
                self.tiles[i + j*self.nrows_tiles].copy_in_vec(&mut other[shift..], lda)
            }
        }
    }
}

impl<T> std::ops::Index<(usize,usize)> for TiledMatrix<T>
where
    T: Copy + Default + std::ops::Sub<Output = T>
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
        assert!(i < self.nrows && j < self.ncols);
        const TILE_SIZE : usize = tile::TILE_SIZE;
        let row_tile = i / TILE_SIZE;
        let col_tile = j / TILE_SIZE;
        let row_in_tile = i - row_tile*TILE_SIZE;
        let col_in_tile = j - col_tile*TILE_SIZE;
        assert!(row_tile < self.nrows_tiles && col_tile < self.ncols_tiles);
        let tile = &self.tiles[ row_tile + col_tile * self.nrows_tiles ];
        &tile[(row_in_tile, col_in_tile)]
    }
}

impl<T> std::ops::IndexMut<(usize,usize)> for TiledMatrix<T>
where
    T: Copy + Default + std::ops::Sub<Output = T>
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
        assert!(i < self.nrows && j < self.ncols);
        const TILE_SIZE : usize = tile::TILE_SIZE;
        let row_tile = i / TILE_SIZE;
        let col_tile = j / TILE_SIZE;
        let row_in_tile = i - row_tile*TILE_SIZE;
        let col_in_tile = j - col_tile*TILE_SIZE;
        assert!(row_tile < self.nrows_tiles && col_tile < self.ncols_tiles);
        let tile = &mut self.tiles[ row_tile + col_tile * self.nrows_tiles ];
        &mut tile[(row_in_tile, col_in_tile)]
    }
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

}
