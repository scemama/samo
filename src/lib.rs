//! # Syncronous/Asynchonous Matrix Operations
//!
//! `samo` is a collection of utilities to work with matrices that are
//! too large to fit into memory efficiently and need to be divided
//! into smaller blocks or "tiles".  This approach can significantly
//! improve performance for operations on matrices by optimizing for
//! cache locality and parallel computation.
//!
//! ## Modules
//!
//! - `tile`: Contains the `Tile` struct and its associated
//!   implementations. A `Tile` is a submatrix block that represents a
//!   portion of a larger matrix.
//! - `tiled_matrix`: Contains the `TiledMatrix` struct that
//!   represents a matrix divided into tiles. This module provides the
//!   necessary functionality to work with tiled matrices as a whole.
//!
//! ## Usage
//!
//! This crate is meant to be used with numeric computations where
//! matrix operations are a performance bottleneck. By dividing a
//! matrix into tiles, operations can be performed on smaller subsets
//! of the data, which can be more cache-friendly and parallelizable.
//!
//! ### Example
//!
//! ```
//! use samo::TiledMatrix;
//!
//! // Create a 64x64 matrix and fill it with ones.
//! let matrix = TiledMatrix::new(64, 64, 1.0);
//!
//! // Perform operations on the matrix...
//! ```

// Expose the tile and tiled_matrix modules to any crate that depends on this one.
pub mod tile;
pub mod tiled_matrix;
pub use tiled_matrix::TiledMatrix;


