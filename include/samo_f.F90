module samo
  use, intrinsic :: iso_c_binding
  implicit none

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
      subroutine samo_sfree(a_tiled) bind(C)
        import
        implicit none
        type(c_ptr), value         :: a_tiled
      end subroutine
   end interface

   interface
      subroutine samo_dfree(a_tiled) bind(C)
        import
        implicit none
        type(c_ptr), value         :: a_tiled
      end subroutine
   end interface

end module samo
