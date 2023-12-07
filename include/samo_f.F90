module samo
  use, intrinsic :: iso_c_binding
  implicit none

   interface
      subroutine samo_set_device(id) bind(C)
        import
        implicit none
        integer(c_int32_t), value        :: id
      end subroutine
   end interface

   interface
      integer(c_int32_t) function samo_get_device() bind(C)
         import
         implicit none
      end function
   end interface

   interface
      integer(c_int32_t) function samo_get_device_count() bind(C)
         import
         implicit none
      end function
   end interface

   interface
      subroutine samo_await(handle) bind(C)
        import
        implicit none
        type(c_ptr), value        :: handle
      end subroutine
   end interface

   interface
      type(c_ptr) function samo_dtile(a, nrows, ncols, lda) bind(C)
        import
        implicit none
        integer(c_int64_t), value  :: nrows, ncols, lda
        real(c_double)             :: a(lda,ncols)
      end function
   end interface

   interface
      type(c_ptr) function samo_stile(a, nrows, ncols, lda) bind(C)
        import
        implicit none
        integer(c_int64_t), value  :: nrows, ncols, lda
        real(c_float)              :: a(lda,ncols)
      end function
   end interface

   interface
      type(c_ptr) function samo_dreshape(a, nrows, ncols) bind(C)
        import
        implicit none
        integer(c_int64_t), value  :: nrows, ncols
        type(c_ptr), value         :: a
      end function
   end interface

   interface
      type(c_ptr) function samo_sreshape(a, nrows, ncols) bind(C)
        import
        implicit none
        integer(c_int64_t), value  :: nrows, ncols
        type(c_ptr), value         :: a
      end function
   end interface

   interface
      subroutine samo_duntile(a_tiled, a, lda) bind(C)
        import
        implicit none
        type(c_ptr), value         :: a_tiled
        integer(c_int64_t), value  :: lda
        real(c_double)             :: a(lda,*)
      end subroutine
   end interface

   interface
      subroutine samo_suntile(a_tiled, a, lda) bind(C)
        import
        implicit none
        type(c_ptr), value         :: a_tiled
        integer(c_int64_t), value  :: lda
        real(c_float)              :: a(lda,*)
      end subroutine
   end interface


   interface
      subroutine samo_dfree(a_tiled) bind(C)
        import
        implicit none
        type(c_ptr), value         :: a_tiled
      end subroutine
   end interface

   interface
      subroutine samo_sfree(a_tiled) bind(C)
        import
        implicit none
        type(c_ptr), value         :: a_tiled
      end subroutine
   end interface


   interface
      subroutine samo_dgemm_tiled(transa, transb, alpha, a, b, beta, c) bind(C)
        import
        implicit none
        character(c_char), value  :: transa, transb
        real(c_double), value     :: alpha, beta
        type(c_ptr), value        :: a, b, c
      end subroutine
   end interface

   interface
      subroutine samo_sgemm_tiled(transa, transb, alpha, a, b, beta, c) bind(C)
        import
        implicit none
        character(c_char), value  :: transa, transb
        real(c_float), value      :: alpha, beta
        type(c_ptr), value        :: a, b, c
      end subroutine
   end interface

   interface
      type(c_ptr) function samo_duntile_async(a_tiled, a, lda) bind(C)
        import
        implicit none
        type(c_ptr), value         :: a_tiled
        integer(c_int64_t), value  :: lda
        real(c_double)             :: a(lda,*)
      end function
   end interface

   interface
      type(c_ptr) function samo_suntile_async(a_tiled, a, lda) bind(C)
        import
        implicit none
        type(c_ptr), value         :: a_tiled
        integer(c_int64_t), value  :: lda
        real(c_float)              :: a(lda,*)
      end function
   end interface


   interface
      type(c_ptr) function samo_dgemm_tiled_async(transa, transb, alpha, a, b, beta, c) bind(C)
        import
        implicit none
        character(c_char), value  :: transa, transb
        real(c_double), value      :: alpha, beta
        type(c_ptr), value        :: a, b, c
      end function
   end interface

   interface
      type(c_ptr) function samo_sgemm_tiled_async(transa, transb, alpha, a, b, beta, c) bind(C)
        import
        implicit none
        character(c_char), value  :: transa, transb
        real(c_float), value      :: alpha, beta
        type(c_ptr), value        :: a, b, c
      end function
   end interface

   ! GPU
   ! ---

   interface
      type(c_ptr) function samo_dtile_gpu(a, nrows, ncols, lda) bind(C)
        import
        implicit none
        integer(c_int64_t), value  :: nrows, ncols, lda
        real(c_double)             :: a(lda,ncols)
      end function
   end interface

   interface
      type(c_ptr) function samo_stile_gpu(a, nrows, ncols, lda) bind(C)
        import
        implicit none
        integer(c_int64_t), value  :: nrows, ncols, lda
        real(c_float)              :: a(lda,ncols)
      end function
   end interface

   interface
      type(c_ptr) function samo_dreshape_gpu(a, nrows, ncols) bind(C)
        import
        implicit none
        integer(c_int64_t), value  :: nrows, ncols
        type(c_ptr), value         :: a
      end function
   end interface

   interface
      type(c_ptr) function samo_sreshape_gpu(a, nrows, ncols) bind(C)
        import
        implicit none
        integer(c_int64_t), value  :: nrows, ncols
        type(c_ptr), value         :: a
      end function
   end interface


   interface
      subroutine samo_duntile_gpu(a_tiled, a, lda) bind(C)
        import
        implicit none
        type(c_ptr), value         :: a_tiled
        integer(c_int64_t), value  :: lda
        real(c_double)             :: a(lda,*)
      end subroutine
   end interface

   interface
      subroutine samo_suntile_gpu(a_tiled, a, lda) bind(C)
        import
        implicit none
        type(c_ptr), value         :: a_tiled
        integer(c_int64_t), value  :: lda
        real(c_float)              :: a(lda,*)
      end subroutine
   end interface


   interface
      subroutine samo_dfree_gpu(a_tiled) bind(C)
        import
        implicit none
        type(c_ptr), value         :: a_tiled
      end subroutine
   end interface

   interface
      subroutine samo_sfree_gpu(a_tiled) bind(C)
        import
        implicit none
        type(c_ptr), value         :: a_tiled
      end subroutine
   end interface


   interface
      subroutine samo_dgemm_tiled_gpu(transa, transb, alpha, a, b, beta, c) bind(C)
        import
        implicit none
        character(c_char), value  :: transa, transb
        real(c_double), value     :: alpha, beta
        type(c_ptr), value        :: a, b, c
      end subroutine
   end interface

   interface
      subroutine samo_sgemm_tiled_gpu(transa, transb, alpha, a, b, beta, c) bind(C)
        import
        implicit none
        character(c_char), value  :: transa, transb
        real(c_float), value      :: alpha, beta
        type(c_ptr), value        :: a, b, c
      end subroutine
   end interface


   interface
      type(c_ptr) function samo_duntile_gpu_async(a_tiled, a, lda) bind(C)
        import
        implicit none
        type(c_ptr), value         :: a_tiled
        integer(c_int64_t), value  :: lda
        real(c_double)             :: a(lda,*)
      end function
   end interface

   interface
      type(c_ptr) function samo_suntile_gpu_async(a_tiled, a, lda) bind(C)
        import
        implicit none
        type(c_ptr), value         :: a_tiled
        integer(c_int64_t), value  :: lda
        real(c_float)              :: a(lda,*)
      end function
   end interface


   interface
      type(c_ptr) function samo_dgemm_tiled_gpu_async(transa, transb, alpha, a, b, beta, c) bind(C)
        import
        implicit none
        character(c_char), value  :: transa, transb
        real(c_double), value      :: alpha, beta
        type(c_ptr), value        :: a, b, c
      end function
   end interface

   interface
      type(c_ptr) function samo_sgemm_tiled_gpu_async(transa, transb, alpha, a, b, beta, c) bind(C)
        import
        implicit none
        character(c_char), value  :: transa, transb
        real(c_float), value      :: alpha, beta
        type(c_ptr), value        :: a, b, c
      end function
   end interface

end module samo
