use crate::cuda;
use crate::cublas;
use crate::tile::Tile;
use crate::tile::TILE_SIZE;

use crate::cuda::DevPtr;

use std::cell::RefCell;

#[derive(Debug,Clone)]
pub(crate) enum Dirt {
    Host, Device, Clean
}
use Dirt::*;

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

    /// A local proxy to the data on the device
    pub(crate) local_data: Option<Vec<T>>,

    /// If true, the data on the device is not in sync with the data in the proxy.
    pub(crate) dirty: RefCell<Dirt>,

    /// The number of rows in the tile, not exceeding `TILE_SIZE`.
    pub(crate) nrows: usize,

    /// The number of columns in the tile, not exceeding `TILE_SIZE`.
    pub(crate) ncols: usize,

    /// Flag to specify if the matrix is transposed. For transposed
    /// matrices, the terms row and column need to be swapped.
    pub(crate) transposed: bool,

    /// Cuda stream used for async operations
    pub(crate) stream: cuda::Stream,

    /// Cublas handle
    pub(crate) cublas: cublas::Context,
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
            pub fn new(cublas: &cublas::Context, nrows: usize, ncols: usize, init: Option<$s>) -> Self {
                assert!(nrows <= TILE_SIZE, "Too many rows: {nrows} > {TILE_SIZE}");
                assert!(ncols <= TILE_SIZE, "Too many columns: {ncols} > {TILE_SIZE}");

                let size = ncols * nrows;
                let dev_ptr = DevPtr::<$s>::malloc(size).unwrap();

                let stream = cuda::Stream::create().unwrap();

                let (local_data, dirty) = match init {
                    Some(init) => { (Some(vec![init ; size]), RefCell::new(Host) ) },
                    None       => { (None              , RefCell::new(Device)) },
                };

                let transposed = false;
                let cublas = cublas.clone();
                Self { cublas, dev_ptr, dirty, local_data, stream, nrows, ncols, transposed}
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
            pub fn from(cublas: &cublas::Context, other: &[$s], nrows: usize, ncols:usize, lda:usize) -> Self {
                assert!(nrows <= TILE_SIZE, "Too many rows: {nrows} > {TILE_SIZE}");
                assert!(ncols <= TILE_SIZE, "Too many columns: {ncols} > {TILE_SIZE}");
                let size = ncols * nrows;
                let mut dev_ptr = DevPtr::malloc(size).unwrap();

                let stream = cuda::Stream::create().unwrap();

                let local_data = None;
                cublas::set_matrix_async(nrows, ncols, other, lda,
                   &mut dev_ptr, nrows, &stream).unwrap();

                let dirty = RefCell::new(Device);
                let transposed = false;
                let cublas = cublas.clone();

                Self { cublas, dev_ptr, dirty, local_data, stream, nrows, ncols, transposed}

            }

            pub fn from_tile(cublas: &cublas::Context, other: &Tile::<$s>) -> Self {
                let nrows = other.nrows;
                let ncols = other.ncols;
                let size = ncols * nrows;
                let mut dev_ptr = DevPtr::malloc(size).unwrap();

                let stream = cuda::Stream::create().unwrap();

                cublas::set_matrix_async(nrows, ncols, &other.data, nrows,
                   &mut dev_ptr, nrows, &stream).unwrap();

                let transposed = other.transposed;
                let local_data = None;

                let dirty = RefCell::new(Device);
                let cublas = cublas.clone();

                Self { cublas, dev_ptr, dirty, local_data, stream, nrows, ncols, transposed}

            }


            #[inline]
            fn dirty(&self) -> Dirt {
                (*(self.dirty.borrow())).clone()
            }

            #[inline]
            pub fn sync_to_device(&self) {
                match self.dirty() {
                    Device | Clean => {},
                    Host => {
                        let mut dev_ptr_out = self.dev_ptr.clone();
                        let local_data = self.local_data.as_ref().unwrap();
                        cublas::set_matrix_async(self.nrows, self.ncols,
                            local_data, self.nrows,
                            &mut dev_ptr_out, self.nrows,
                            &self.stream).unwrap();
                        self.stream.synchronize().unwrap();
                        self.update_dirty(Clean);
                    }
                }
            }

            #[inline]
            pub fn sync_from_device(&mut self) {
                match self.dirty() {
                    Host | Clean => { () },
                    Device => {
                        match self.local_data {
                           None => { self.local_data = Some(vec![0. ; self.nrows*self.ncols]) }
                           _ => { () }
                        };
                        cublas::get_matrix_async(self.nrows, self.ncols,
                            &self.dev_ptr, self.nrows,
                            &mut self.local_data.as_mut().unwrap(), self.nrows,
                            &self.stream).unwrap();
                        self.stream.synchronize().unwrap();
                        self.update_dirty(Clean);
                    }
                }
            }

            #[inline]
            fn update_dirty(&self, status: Dirt) {
                *self.dirty.borrow_mut() = status;
            }

            #[inline]
            fn update_dirty_mut(&mut self, status: Dirt) {
                *self.dirty.borrow_mut() = status;
            }

            #[inline]
            pub fn data(&mut self) -> &[$s] {
                self.sync_from_device();
                &self.local_data.as_ref().unwrap()
            }

            /// Copies the tile's data into a provided mutable slice,
            /// effectively transforming the tile's internal one-dimensional
            /// representation back into a two-dimensional array layout.
            ///
            /// # Arguments
            ///
            /// * `other` - The mutable slice into which the tile's data will be copied.
            /// * `lda` - The leading dimension to be used when copying the data into `other`.
            pub fn copy_in_vec(&mut self, other: &mut [$s], lda:usize) {
                self.sync_from_device();
                let local_data = self.local_data.as_ref().unwrap();
                match self.transposed {
                    false => {
                        for j in 0..self.ncols {
                            let shift_tile = j*self.nrows;
                            let shift_array = j*lda;
                            other[shift_array..(self.nrows + shift_array)].copy_from_slice(&local_data[shift_tile..(self.nrows + shift_tile)]);
                        }},
                    true => {
                        for i in 0..self.nrows {
                            let shift_array = i*lda;
                            for j in 0..self.ncols {
                                let shift_tile = j*self.nrows;
                                other[j + shift_array] = local_data[i + shift_tile];
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
            pub unsafe fn get_unchecked(&mut self, i:usize, j:usize) -> &$s {
                self.sync_from_device();
                let local_data = self.local_data.as_ref().unwrap();
                match self.transposed {
                    false => unsafe { local_data.get_unchecked(i + j * self.nrows) },
                    true  => unsafe { local_data.get_unchecked(j + i * self.nrows) },
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
                self.sync_from_device();
                let local_data = self.local_data.as_mut().unwrap();
                match self.transposed {
                    false => unsafe { local_data.get_unchecked_mut(i + j * self.nrows) },
                    true  => unsafe { local_data.get_unchecked_mut(j + i * self.nrows) },
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
                let size = self.ncols*self.nrows;
                let mut x : Vec::<$s> = vec![ 0. ; size ];
                let data : &[$s] =
                  match *self.dirty.borrow() {
                    Device => {
                        cublas::get_matrix_async(self.nrows, self.ncols,
                            &self.dev_ptr, self.nrows,
                            &mut x, self.nrows,
                            &self.stream).unwrap();
                        self.stream.synchronize().unwrap();
                        x.as_slice()
                    }
                    _ => { self.local_data.as_ref().unwrap() }
                  };
                let mut new_tile = Self::from(&self.cublas, data,
                     self.nrows, self.ncols, self.nrows);
                new_tile.transposed = !self.transposed;
                new_tile
            }

            /// Rescale the tile
            #[inline]
            pub fn scale_mut(&mut self, factor: $s) {
                self.stream.set_active(&self.cublas).unwrap();

                let dev_ptr_in = self.dev_ptr.clone();

                self.sync_to_device();

                $geam(&self.cublas, b'N', b'N', self.nrows, self.ncols,
                  factor, &dev_ptr_in, self.nrows, 0., &dev_ptr_in, self.nrows,
                  &mut self.dev_ptr, self.nrows).unwrap();

                self.update_dirty(Device);
            }


            /// Returns a copy of the tile, rescaled
            #[inline]
            pub fn scale(&self, factor: $s) -> Self {

                let mut other = Self::new(&self.cublas, self.nrows, self.ncols, None);

                other.stream.set_active(&other.cublas).unwrap();

                self.sync_to_device();

                $geam(&self.cublas, b'N', b'N', self.nrows, self.ncols,
                  factor, &self.dev_ptr, self.nrows, 0., &self.dev_ptr, self.nrows,
                  &mut other.dev_ptr, other.nrows).unwrap();

                other.update_dirty(Device);
                other
            }

            /// Add another tile to the tile
            #[inline]
            pub fn add_mut(&mut self, other: &Self) {
                assert_eq!(self.ncols(), other.ncols());
                assert_eq!(self.nrows(), other.nrows());

                let transa = b'N';
                let transb = if other.transposed() != self.transposed() { b'T' } else { b'N' };
                let dev_ptr_in = self.dev_ptr.clone();

                self.stream.set_active(&self.cublas).unwrap();

                other.sync_to_device();
                self.sync_to_device();

                $geam(&self.cublas, transa, transb, self.nrows(), self.ncols(),
                  1., &dev_ptr_in, self.nrows, 1., &other.dev_ptr, other.nrows,
                  &mut self.dev_ptr, self.nrows).unwrap();

                self.update_dirty(Device);
            }

            /// Adds another tile to the tile and returns a new tile with the result.
            #[inline]
            pub fn add(&self, other: &Self) -> Self {
                assert_eq!(self.ncols(), other.ncols());
                assert_eq!(self.nrows(), other.nrows());

                let mut result = Self::new(&self.cublas, self.nrows, self.ncols, None);
                result.stream = self.stream.clone();

                let transa = if self.transposed()  { b'T' } else { b'N' };
                let transb = if other.transposed() { b'T' } else { b'N' };

                result.stream.set_active(&self.cublas).unwrap();

                self.sync_to_device();
                other.sync_to_device();

                $geam(&self.cublas, transa, transb, self.nrows(), self.ncols(),
                  1., &self.dev_ptr, self.nrows, 1., &other.dev_ptr, other.nrows,
                  &mut result.dev_ptr, result.nrows).unwrap();

                result.update_dirty(Device);
                result
            }


            pub fn geam_mut (alpha: $s, a: &Self, beta: $s, b: &Self, c: &mut Self)
            {
                assert_eq!(a.ncols(), b.ncols());
                assert_eq!(a.ncols(), c.ncols());
                assert_eq!(a.nrows(), b.nrows());
                assert_eq!(a.nrows(), c.nrows());
                assert!(!c.transposed);

                let transa = if a.transposed { b'T' } else { b'N' };
                let transb = if b.transposed { b'T' } else { b'N' };

                a.sync_to_device();
                b.sync_to_device();
                c.sync_to_device();

                c.stream.set_active(&c.cublas).unwrap();

                $geam(&c.cublas, transa, transb, c.nrows(), c.ncols(),
                  alpha, &a.dev_ptr, a.nrows, beta, &b.dev_ptr, b.nrows,
                  &mut c.dev_ptr, c.nrows).unwrap();

                c.update_dirty(Device);
            }

            pub fn geam (alpha: $s, a: &Self, beta: $s, b: &Self) -> Self
            {
                let mut c = Self::new(&a.cublas, a.nrows(), a.ncols(), None);
                Self::geam_mut(alpha, a, beta, b, &mut c);
                c
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

                a.sync_to_device();
                b.sync_to_device();
                c.sync_to_device();

                c.stream.set_active(&c.cublas).unwrap();

                $gemm(&c.cublas, transa, transb, m, n, k, alpha,
                  &a.dev_ptr, lda, &b.dev_ptr, ldb, beta,
                  &mut c.dev_ptr, ldc).unwrap();

                c.update_dirty(Device);

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
            pub fn gemm (alpha: $s, a: &Self, b: &Self) -> Self
            {
                let mut c = Self::new(&a.cublas, a.nrows(), b.ncols(), None);
                Self::gemm_mut(alpha, a, b, 0.0, &mut c);
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
                match self.dirty() {
                    Device => panic!("Device is dirty. Run sync_from_device() first."),
                    _ => ()
                };
                let local_data = self.local_data.as_ref().unwrap();
                match self.transposed {
                    false => { assert!(i < self.nrows && j < self.ncols); &local_data[i + j * self.nrows] },
                    true  => { assert!(j < self.nrows && i < self.ncols); &local_data[j + i * self.nrows] },
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
                self.sync_from_device();
                self.update_dirty_mut(Host);
                  match self.transposed {
                    false => { assert!(i < self.nrows && j < self.ncols);
                            &mut self.local_data.as_mut().unwrap()[i + j * self.nrows] },
                    true  => { assert!(j < self.nrows && i < self.ncols);
                            &mut self.local_data.as_mut().unwrap()[j + i * self.nrows] },
                  }
            }
        }
    }
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
                let handle = cublas::Context::new().unwrap();
                let tile = TileGPU::<$s>::new(&handle, 10, 20, Some(1.0));
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

                let mut tile = TileGPU::<$s>::from(&handle, &other, 10, 15, 20);

                let mut other = vec![0.0 ; 150];
                tile.copy_in_vec(&mut other, 10);
                assert_eq!(other, other_ref);

                let tile_cpu = Tile::<$s>::from(&other, 10, 15, 10);
                let mut tile_gpu = TileGPU::<$s>::from_tile(&handle, &tile_cpu);
                assert_eq!(tile_cpu.nrows(), tile_gpu.nrows());
                assert_eq!(tile_cpu.ncols(), tile_gpu.ncols());
                assert_eq!(tile_cpu.transposed(), tile_gpu.transposed());
                tile_gpu.sync_from_device();
                for j in 0..(tile.ncols()) {
                    for i in 0..(tile.nrows()) {
                        assert_eq!(tile_cpu[(i,j)], tile_gpu[(i,j)]);
                    }
                }
            }

            #[test]
            #[should_panic]
            fn $creation_too_large() {
                let handle = cublas::Context::new().unwrap();
                let _ = TileGPU::<$s>::new(&handle, TILE_SIZE+1, 10, Some(1.0));
            }

            #[test]
            #[should_panic]
            fn $row_overflow() {
                let handle = cublas::Context::new().unwrap();
                let tile = TileGPU::<$s>::new(&handle, 10, 20, Some(1.0));
                let _ = tile[(11,10)];
            }

            #[test]
            #[should_panic]
            fn $col_overflow() {
                let handle = cublas::Context::new().unwrap();
                let tile = TileGPU::<$s>::new(&handle, 10, 20, Some(1.0));
                let _ = tile[(1,21)];
            }

            #[test]
            fn $index_mut() {
                let handle = cublas::Context::new().unwrap();
                let mut tile = TileGPU::<$s>::new(&handle, 10, 20, Some(1.0));
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
                let handle = cublas::Context::new().unwrap();
                let mut tile = TileGPU::<$s>::new(&handle, 10, 20, Some(1.0));
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
                let handle = cublas::Context::new().unwrap();
                let mut tile = TileGPU::<$s>::new(&handle, 10, 20, Some(1.0));
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
                let mut new_tile = tile.scale(2.0);
                assert_eq!(new_tile.data(), data_ref);

                tile.scale_mut(2.0);
                assert_eq!(tile.data(), data_ref);
            }

            #[test]
            #[should_panic]
            fn $geam_wrong_size() {
                let handle = cublas::Context::new().unwrap();
                let n = 10;
                let m = 5;
                let a = TileGPU::<$s>::new(&handle, m, n, Some(0.0));
                let b = TileGPU::<$s>::new(&handle, n, m, Some(0.0));
                let _ = TileGPU::<$s>::geam(1.0, &a, 1.0, &b);
            }

            #[test]
            fn $geam_cases() {
                let handle = cublas::Context::new().unwrap();
                let n = 8;
                let m = 4;
                let mut a   = TileGPU::<$s>::new(&handle, m, n, Some(0.0));
                let mut a_t = TileGPU::<$s>::new(&handle, n, m, Some(0.0));
                let mut b   = TileGPU::<$s>::new(&handle, m, n, Some(0.0));
                let mut b_t = TileGPU::<$s>::new(&handle, n, m, Some(0.0));
                let mut zero_tile = TileGPU::<$s>::new(&handle, m, n, Some(0.0));
                let mut zero_tile_t = TileGPU::<$s>::new(&handle, n, m, Some(0.0));
                for i in 0..m {
                    for j in 0..n {
                        a[(i,j)] = (i * 10 + j) as $s;
                        b[(i,j)] = (i * 1000 + j*10) as $s;
                        a_t[(j,i)] = (i * 10 + j) as $s;
                        b_t[(j,i)] = (i * 1000 + j*10) as $s;
                    }
                }

                assert_eq!(
                    TileGPU::<$s>::geam(0.0, &a, 0.0, &b).data(),
                    zero_tile.data());

                assert_eq!(
                    TileGPU::<$s>::geam(1.0, &a, 0.0, &b).data(),
                    a.data());

                assert_eq!(
                    TileGPU::<$s>::geam(0.0, &a, 1.0, &b).data(),
                    b.data());

                let mut r = TileGPU::<$s>::new(&handle, m, n, Some(0.0));
                for i in 0..m {
                  for j in 0..n {
                    r[(i,j)] = 2.0*a[(i,j)];
                  }
                };
                assert_eq!( TileGPU::<$s>::geam(2.0, &a, 0.0, &b).data(),
                      r.data() );

                assert_eq!(
                    TileGPU::<$s>::geam(0.5, &a, 0.5, &a).data(),
                    a.data());

                assert_eq!(
                    TileGPU::<$s>::geam(0.5, &a, -0.5, &a).data(),
                    zero_tile.data());

                assert_eq!(
                    TileGPU::<$s>::geam(1.0, &a, -1.0, &a).data(),
                    zero_tile.data());

                let mut r = TileGPU::<$s>::new(&handle, m, n, Some(0.0));
                for i in 0..m {
                  for j in 0..n {
                    r[(i,j)] = 2.0*b[(i,j)];
                  }
                };
                assert_eq!( TileGPU::<$s>::geam(0.0, &a, 2.0, &b).data(),
                      r.data().clone() );

                let mut r = TileGPU::<$s>::new(&handle, m, n, Some(0.0));
                for i in 0..m {
                  for j in 0..n {
                    r[(i,j)] = a[(i,j)] + b[(i,j)];
                  }
                };
                assert_eq!( TileGPU::<$s>::geam(1.0, &a, 1.0, &b).data(),
                      r.data() );

                let mut r = TileGPU::<$s>::new(&handle, m, n, Some(0.0));
                for i in 0..m {
                    for j in 0..n {
                        r[(i,j)] = b[(i,j)] - a[(i,j)];
                    }
                };
                assert_eq!( TileGPU::<$s>::geam(-1.0, &a, 1.0, &b).data(),
                      r.data() );

                let mut r = TileGPU::<$s>::new(&handle, m, n, Some(0.0));
                for i in 0..m {
                    for j in 0..n {
                        r[(i,j)] = a[(i,j)] - b[(i,j)];
                    }
                };
                assert_eq!( TileGPU::<$s>::geam(1.0, &a, -1.0, &b).data(),
                      r.data());

                assert_eq!(
                    TileGPU::<$s>::geam(1.0, &a, -1.0, &a_t.t()).data(),
                    zero_tile.data());

                assert_eq!(
                    TileGPU::<$s>::geam(1.0, &a_t.t(), -1.0, &a).data(),
                    zero_tile_t.data());


                // Mutable geam

                let mut c1 = TileGPU::<$s>::geam(1.0, &a, 1.0, &b);
                let mut c  = TileGPU::<$s>::geam(1.0, &a, 1.0, &b);
                TileGPU::<$s>::geam_mut(-1.0, &a, 1.0, &c1, &mut c);
                assert_eq!( c.data() , b.data() );

                for (alpha, beta) in [ (1.0,1.0), (1.0,-1.0), (-1.0,-1.0), (-1.0,1.0),
                                       (1.0,0.0), (0.0,-1.0), (0.5, 1.0), (0.5, 1.0),
                                       (0.5,-0.5) ] {
                        let nrows = a.nrows();
                        let ncols = a.ncols();
                        let mut c = TileGPU::<$s>::from(&handle, a.data(), nrows,
                              ncols, nrows);
                        TileGPU::<$s>::geam_mut(alpha, &a, beta, &b, &mut c);
                        assert_eq!( c.data(),
                            TileGPU::<$s>::geam(alpha, &a, beta, &b).data());
                                       }
            }

            #[test]
            fn $gemm() {
                let handle = cublas::Context::new().unwrap();
                let a = TileGPU::<$s>::from(&handle,  &[1. , 1.1, 1.2, 1.3,
                                    2. , 2.1, 2.2, 2.3,
                                    3. , 3.1, 3.2, 3.3],
                                    4, 3, 4).t();

                let b = TileGPU::<$s>::from(&handle,  &[ 1.0, 2.0, 3.0, 4.0,
                                    1.1, 2.1, 3.1, 4.1],
                                    4, 2, 4 );

                let c_ref = TileGPU::<$s>::from(&handle,  &[12.0 , 22.0 , 32.0,
                                        12.46, 22.86, 33.26],
                                        3, 2, 3);

                let mut c = TileGPU::<$s>::gemm(1.0, &a, &b);
                let mut difference = TileGPU::<$s>::geam(1.0, &c, -1.0, &c_ref);
                c.sync_from_device();
                difference.sync_from_device();
                for j in 0..2 {
                    for i in 0..3 {
                        assert!(num::abs(difference[(i,j)] / c[(i,j)]) < <$s>::EPSILON);
                    }
                }

                let a = a.t();
                let b = b.t();
                let mut c_t = TileGPU::<$s>::gemm(1.0, &b, &a);
                let mut difference = TileGPU::<$s>::geam(1.0, &c_t, -1.0, &c.t());
                c_t.sync_from_device();
                difference.sync_from_device();
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



