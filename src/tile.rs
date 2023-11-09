/// A constant representing the leading dimension of arrays in tiles,
/// which is also the maximum number of rows and columns a `Tile` can
/// have.
pub const TILE_SIZE: usize = 32;

/// A `Tile` is a data structure that represents a dense block of a
/// matrix, often used in block matrix operations to optimize for
/// cache usage, parallelism, and memory bandwidth.
/// 
/// Generic over `T` which is the type of elements stored in the
/// tile. It requires `T` to be `Copy` and have a `Default` value, 
/// to allow for easy initialization.
#[derive(Debug)]
pub struct Tile<T>
where
    T: Copy + Default
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
    T: Copy + Default
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
}

/// Implementation of the Index trait to allow for read access to
/// elements in the Tile using array indexing syntax.
impl<T> std::ops::Index<(usize,usize)> for Tile<T>
where
    T: Copy + Default + std::ops::Sub<Output = T>
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
    T: Copy + Default + std::ops::Sub<Output = T>
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

}
