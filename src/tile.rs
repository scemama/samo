const LDA : usize = 512;

/// Data structure for a tile
pub struct Tile<T> {
    /// Data stored in the tile
    data: Vec<T>,

    /// Number of rows
    nrows : usize,

    /// Number of columns
    ncols : usize,
}


impl<T> Tile<T>
where
    T: Copy + Default + std::ops::Add<Output = T>,
{
    /// Creates a new nrows x ncols Tile
    pub fn new(nrows:usize, ncols:usize, init:T) -> Self {
        if nrows > LDA {
            panic!("Too many rows");
        }
        if ncols > LDA {
            panic!("Too many columns");
        }
        let size = ncols * LDA;
        let mut data = Vec::<T>::with_capacity(size);
        for _ in 0..size {
            data.push(init);
        }
        Tile { data, nrows, ncols }
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

    /// A method to add two tiles and return a new tile.
    pub fn add(&self, other: &Self) -> Self {
        assert!(self.ncols == other.ncols && self.nrows == other.nrows,
                "Dimensions don;t match");
        let nrows = self.nrows;
        let ncols = self.ncols;
        let size = ncols * LDA;
        let mut data = Vec::<T>::with_capacity(size);
        for i in 0..size {
            data.push( self.data[i] + other.data[i] );
        }
        Tile { data, nrows, ncols }
    }
}

impl<T> std::ops::Index<(usize,usize)> for Tile<T> {
     type Output = T;
     fn index(&self, (i,j): (usize,usize)) -> &Self::Output {
         assert!(i < self.nrows && j < self.ncols);
         &self.data[i + j * LDA]
     }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creation() {
        let tile = Tile::new(10, 20, 1.0);
        assert_eq!(tile.nrows, 10);
        assert_eq!(tile.ncols, 20);
        for j in 0..(tile.ncols) {
            for i in 0..(tile.nrows) {
                assert_eq!(tile[(i,j)], 1.0);
            }
        }
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

}
