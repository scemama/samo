use crate::tile;
use crate::tile::Tile;

#[derive(Debug)]
pub struct TiledMatrix<T>
where
    T: Copy + Default + std::ops::Sub<Output = T>
{
    /// Number of rows
    nrows: usize,

    /// Number of columns
    ncols: usize,

    /// Vector of tiles
    tiles: Vec<Tile<T>>,

    /// Number of rows of tiles
    nrows_tiles: usize,

    /// Number of columns of tiles
    ncols_tiles: usize,
}

impl<T> TiledMatrix<T>
where
    T: Copy + Default + std::ops::Sub<Output = T>
{
    pub fn new(nrows: usize, ncols: usize, init: T) -> Self {
        let lda = tile::LDA;
        let nrows_tiles_full = nrows / lda;
        let ncols_tiles_full = ncols / lda;

        let nrows_tiles = 
            if nrows_tiles_full * lda < nrows {
                nrows_tiles_full + 1
            } else {
                nrows_tiles_full
            };

        let ncols_tiles = 
            if ncols_tiles_full * lda < ncols {
                ncols_tiles_full + 1
            } else {
                ncols_tiles_full
            };

        let nrows_border = nrows - nrows_tiles_full * lda;
        let ncols_border = ncols - ncols_tiles_full * lda;
        let size = nrows_tiles * ncols_tiles;
        let mut tiles = Vec::<Tile<T>>::with_capacity(size);
        for _ in 0..ncols_tiles_full {
            for _ in 0..nrows_tiles_full {
                tiles.push( Tile::<T>::new(lda,lda,init) );
            }
            if nrows_tiles > nrows_tiles_full { 
                tiles.push( Tile::<T>::new(nrows_border,lda,init) );
            }
        }
        if ncols_tiles > ncols_tiles_full {
            for _ in 0..nrows_tiles_full {
                tiles.push( Tile::<T>::new(lda,ncols_border,init) );
            }
            if nrows_tiles > nrows_tiles_full { 
                tiles.push( Tile::<T>::new(nrows_border,ncols_border,init) );
            }
        }
        TiledMatrix {
            nrows, ncols, tiles, nrows_tiles, ncols_tiles
        }
    }

    /*
    /// Creates a tiled matrix from a matrix
    pub fn from(other: &[T], nrows: usize, lda:usize, ncols: usize) -> Self {
        
    }
    */
}

impl<T> std::ops::Index<(usize,usize)> for TiledMatrix<T>
where
    T: Copy + Default + std::ops::Sub<Output = T>
{
     type Output = T;
     fn index(&self, (i,j): (usize,usize)) -> &Self::Output {
         assert!(i < self.nrows && j < self.ncols);
         let lda = tile::LDA;
         let row_tile = i / lda;
         let col_tile = j / lda;
         let row_in_tile = i - row_tile*lda;
         let col_in_tile = j - col_tile*lda;
         assert!(row_tile < self.nrows_tiles && col_tile < self.ncols_tiles);
         let tile = &self.tiles[ row_tile + col_tile * self.nrows_tiles ];
         &tile[(row_in_tile, col_in_tile)]
     }
}

impl<T> std::ops::IndexMut<(usize,usize)> for TiledMatrix<T>
where
    T: Copy + Default + std::ops::Sub<Output = T>
{
     fn index_mut(&mut self, (i,j): (usize,usize)) -> &mut Self::Output {
         assert!(i < self.nrows && j < self.ncols);
         let lda = tile::LDA;
         let row_tile = i / lda;
         let col_tile = j / lda;
         let row_in_tile = i - row_tile*lda;
         let col_in_tile = j - col_tile*lda;
         assert!(row_tile < self.nrows_tiles && col_tile < self.ncols_tiles);
         let tile = &mut self.tiles[ row_tile + col_tile * self.nrows_tiles ];
         &mut tile[(row_in_tile, col_in_tile)]
     }
}



#[cfg(test)]
mod tests {
    use super::*;

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
    }

    #[test]
    fn number_of_tiles() {
        let lda = tile::LDA;
        let matrix = TiledMatrix::new(2*lda, 10, 0.);
        assert_eq!(matrix.nrows_tiles, 2);
        let matrix = TiledMatrix::new(2*lda+1, 10, 0.);
        assert_eq!(matrix.nrows_tiles, 3);
        let matrix = TiledMatrix::new(10, 2*lda, 0.);
        assert_eq!(matrix.ncols_tiles, 2);
        let matrix = TiledMatrix::new(10, 2*lda+1, 0.);
        assert_eq!(matrix.ncols_tiles, 3);
    }

}
