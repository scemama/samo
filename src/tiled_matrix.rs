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
        const LDA : usize = tile::LDA;
        let nrows_tiles_full = nrows / LDA;
        let ncols_tiles_full = ncols / LDA;

        let nrows_tiles = 
            if nrows_tiles_full * LDA < nrows {
                nrows_tiles_full + 1
            } else {
                nrows_tiles_full
            };

        let ncols_tiles = 
            if ncols_tiles_full * LDA < ncols {
                ncols_tiles_full + 1
            } else {
                ncols_tiles_full
            };

        let nrows_border = nrows - nrows_tiles_full * LDA;
        let ncols_border = ncols - ncols_tiles_full * LDA;
        let size = nrows_tiles * ncols_tiles;
        let mut tiles = Vec::<Tile<T>>::with_capacity(size);
        for _ in 0..ncols_tiles_full {
            for _ in 0..nrows_tiles_full {
                tiles.push( Tile::<T>::new(LDA,LDA,init) );
            }
            if nrows_tiles > nrows_tiles_full { 
                tiles.push( Tile::<T>::new(nrows_border,LDA,init) );
            }
        }
        if ncols_tiles > ncols_tiles_full {
            for _ in 0..nrows_tiles_full {
                tiles.push( Tile::<T>::new(LDA,ncols_border,init) );
            }
            if nrows_tiles > nrows_tiles_full { 
                tiles.push( Tile::<T>::new(nrows_border,ncols_border,init) );
            }
        }
        TiledMatrix {
            nrows, ncols, tiles, nrows_tiles, ncols_tiles
        }
    }


    /// Creates a tiled matrix from a matrix with a leading dimension `lda`.
    pub fn from(other: &[T], nrows: usize, ncols: usize, lda: usize) -> Self {
        const LDA : usize = tile::LDA;
        let nrows_tiles_full = nrows / LDA;
        let ncols_tiles_full = ncols / LDA;

        let nrows_tiles = 
            if nrows_tiles_full * LDA < nrows {
                nrows_tiles_full + 1
            } else {
                nrows_tiles_full
            };

        let ncols_tiles = 
            if ncols_tiles_full * LDA < ncols {
                ncols_tiles_full + 1
            } else {
                ncols_tiles_full
            };

        let nrows_border = nrows - nrows_tiles_full * LDA;
        let ncols_border = ncols - ncols_tiles_full * LDA;
        let size = nrows_tiles * ncols_tiles;
        let mut tiles = Vec::<Tile<T>>::with_capacity(size);
        for j in 0..ncols_tiles_full {
            let ncols_past = j*LDA;
            let elts_from_prev_columns = ncols_past*lda;
            for i in 0..nrows_tiles_full {
                let elts_from_prev_rows    = i*LDA;
                let shift = elts_from_prev_rows + elts_from_prev_columns;
                tiles.push( Tile::<T>::from(&other[shift..], LDA, LDA, lda) );
            }
            if nrows_tiles > nrows_tiles_full { 
                let shift = nrows_tiles_full*LDA + elts_from_prev_columns;
                tiles.push( Tile::<T>::from(&other[shift..], nrows_border, LDA, lda) );
            }
        }
        if ncols_tiles > ncols_tiles_full {
            let ncols_past = ncols_tiles_full*LDA;
            let elts_from_prev_columns = ncols_past*lda;
            for i in 0..nrows_tiles_full {
                let elts_from_prev_rows = i*LDA;
                let shift = elts_from_prev_rows + elts_from_prev_columns;
                tiles.push( Tile::<T>::from(&other[shift..], LDA, ncols_border, lda) );
            }
            if nrows_tiles > nrows_tiles_full { 
                let elts_from_prev_rows = nrows_tiles_full*LDA;
                let shift = elts_from_prev_rows + elts_from_prev_columns;
                tiles.push( Tile::<T>::from(&other[shift..], nrows_border, ncols_border, lda) );
            }
        }
        TiledMatrix {
            nrows, ncols, tiles, nrows_tiles, ncols_tiles
        }
    }

    /// Copy the tiled matrix into a two-dimensional array
    pub fn copy_in_vec(&self, other: &mut [T], lda:usize) {
        const LDA : usize = tile::LDA;

        for j in 0..self.ncols_tiles {
            let elts_from_prev_columns = j*LDA*lda;
            for i in 0..self.nrows_tiles {
                let elts_from_prev_rows = i*LDA;
                let shift = elts_from_prev_rows + elts_from_prev_columns;
                self.tiles[i + j*self.ncols_tiles].copy_in_vec(&mut other[shift..], lda)
            }
        }
    }
}

impl<T> std::ops::Index<(usize,usize)> for TiledMatrix<T>
where
    T: Copy + Default + std::ops::Sub<Output = T>
{
     type Output = T;
     fn index(&self, (i,j): (usize,usize)) -> &Self::Output {
         assert!(i < self.nrows && j < self.ncols);
         const LDA : usize = tile::LDA;
         let row_tile = i / LDA;
         let col_tile = j / LDA;
         let row_in_tile = i - row_tile*LDA;
         let col_in_tile = j - col_tile*LDA;
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
         const LDA : usize = tile::LDA;
         let row_tile = i / LDA;
         let col_tile = j / LDA;
         let row_in_tile = i - row_tile*LDA;
         let col_in_tile = j - col_tile*LDA;
         assert!(row_tile < self.nrows_tiles && col_tile < self.ncols_tiles);
         let tile = &mut self.tiles[ row_tile + col_tile * self.nrows_tiles ];
         &mut tile[(row_in_tile, col_in_tile)]
     }
}



//--------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const LDA : usize = tile::LDA;

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

        let mut other_ref = vec![ 0. ; 65*66 ];
        let mut other = vec![ 0. ; 67*66 ];
        for j in 0..66 {
            for i in 0..65 {
                other_ref[i + j*65] = (i as f64) + (j as f64)*1000.0;
                other[i + j*67] = (i as f64) + (j as f64)*1000.0;
            }
        }
        let matrix = TiledMatrix::from(&other, 65, 66, 67);
        let mut converted = vec![0.0 ; 65*66];
        matrix.copy_in_vec(&mut converted, 65);
        assert_eq!(converted, other_ref);

        for j in 0..66 {
            for i in 0..65 {
                assert_eq!(other[i + j*67], matrix[(i,j)]);
            }
        }
    }

    #[test]
    fn number_of_tiles() {
        let matrix = TiledMatrix::new(2*LDA, 10, 0.);
        assert_eq!(matrix.nrows_tiles, 2);
        let matrix = TiledMatrix::new(2*LDA+1, 10, 0.);
        assert_eq!(matrix.nrows_tiles, 3);
        let matrix = TiledMatrix::new(10, 2*LDA, 0.);
        assert_eq!(matrix.ncols_tiles, 2);
        let matrix = TiledMatrix::new(10, 2*LDA+1, 0.);
        assert_eq!(matrix.ncols_tiles, 3);
    }

}
