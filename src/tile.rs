/// Leading dimension of arrays in tiles. It is also the maximum number of rows and columns.
pub const LDA : usize = 32;

#[derive(Debug)]
/// Data structure for a tile
pub struct Tile<T>
where
    T: Copy + Default + std::ops::Sub<Output = T>
{
    /// Data stored in the tile
    data: Vec<T>,

    /// Number of rows
    nrows: usize,

    /// Number of columns
    ncols: usize,
}


impl<T> Tile<T>
where
    T: Copy + Default + std::ops::Sub<Output = T>
{
    /// Creates a new nrows x ncols Tile
    pub fn new(nrows: usize, ncols: usize, init: T) -> Self {
        if nrows > LDA {panic!("Too many rows");}
        if ncols > LDA {panic!("Too many columns");}
        let size = ncols * LDA;
        let mut data = Vec::<T>::with_capacity(size);
        for _ in 0..ncols {
            for _ in 0..nrows {
                data.push(init);
            }
            for _ in nrows..LDA {
                data.push(init - init);
            }
        }
        Tile { data, nrows, ncols }
    }

    /// Creates a tile from a sub-array
    pub fn from(other: &[T], nrows: usize, ncols:usize, lda:usize) -> Self {
        if nrows > LDA {panic!("Too many rows");}
        if ncols > LDA {panic!("Too many columns");}
        let size = ncols * LDA;
        let mut data = Vec::<T>::with_capacity(size);
        for j in 0..ncols {
            for i in 0..nrows {
                data.push(other[i + j*lda]);
            }
            for _ in nrows..LDA {
                data.push(other[0]-other[0]);
            }
        }
        Tile { data, nrows, ncols }
    }

    /// Copy the tile into a two-dimensional array
    pub fn copy_in_vec(&self, other: &mut [T], lda:usize) {
        for j in 0..self.ncols {
            let shift_tile = j*LDA;
            let shift_array = j*lda;
            for i in 0..self.nrows {
                other[i + shift_array] = self.data[i + shift_tile];
            }
        }
    }

    /// Access an element in the Tile at the given (row, column).
    /// Panics if the index is out of bounds.
    pub fn get(&self, i:usize, j:usize) -> T {
        assert!(i < self.nrows && j < self.ncols, "Index out of bounds");
        self.data[ i + j * LDA ]
    }

    /// Set the value of an element in the Tile at the given (row, column).
    /// Panics if the index is out of bounds.
    pub fn set(&mut self, i:usize, j:usize, value:T) {
        assert!(i < self.nrows && j < self.ncols, "Index out of bounds");
        self.data[ i + j * LDA ] = value;
    }

    /// Returns the number of rows
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Returns the number of columns
    pub fn ncols(&self) -> usize {
        self.ncols
    }
}

impl<T> std::ops::Index<(usize,usize)> for Tile<T>
where
    T: Copy + Default + std::ops::Sub<Output = T>
{
     type Output = T;
     fn index(&self, (i,j): (usize,usize)) -> &Self::Output {
         assert!(i < self.nrows && j < self.ncols);
         &self.data[i + j * LDA]
     }
}

impl<T> std::ops::IndexMut<(usize,usize)> for Tile<T>
where
    T: Copy + Default + std::ops::Sub<Output = T>
{
     fn index_mut(&mut self, (i,j): (usize,usize)) -> &mut Self::Output {
         assert!(i < self.nrows && j < self.ncols);
         &mut self.data[i + j * LDA]
     }
}

#[cfg(test)]
mod tests {
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
        let _ = Tile::new(LDA+1, 10, 1.0);
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
        let mut ref_val = vec![0. ; 20*LDA];
        for j in 0..20 {
            for i in 0..10 {
                ref_val[i + j*LDA] = (i as f64) * 100.0 + (j as f64);
            }
        }
        assert_eq!(tile.data, ref_val);
        
    }

}
