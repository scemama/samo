module samo
  use, intrinsic :: iso_c_binding
  implicit none

  type samo_dmatrix
     type(c_ptr) :: p
     double precision, pointer :: m(:,:)
  end type samo_dmatrix

  interface
     integer(c_int32_t) function samo_get_device_count() bind(C)
       import
       implicit none
     end function samo_get_device_count
  end interface
  
  interface
     subroutine samo_await(handle) bind(C)
       import
       implicit none
       type(c_ptr), value        :: handle
     end subroutine samo_await
  end interface
  
  interface
     function samo_dmalloc_c(device, nrows, ncols) result(ptr) bind(C, name='samo_dmalloc')
       import
       implicit none
       integer(c_int32_t), value  :: device
       integer(c_int64_t), value  :: nrows, ncols
       type(c_ptr)         :: ptr
     end function samo_dmalloc_c
  end interface

  interface
     function samo_dget_pointer_c(a) result(ptr) bind(C, name='samo_dget_pointer')
       import
       implicit none
       type(c_ptr), value  :: a
       type(c_ptr)         :: ptr
     end function samo_dget_pointer_c
  end interface

  interface
     subroutine samo_dfree_c(p) bind(C, name='samo_dfree')
       import
       implicit none
       type(c_ptr), value :: p
     end subroutine samo_dfree_c
  end interface

  interface
     subroutine samo_dgemm_c(transa, transb, alpha, a, b, beta, c) bind(C, name='samo_dgemm')
       import
       implicit none
       character(c_char), value :: transa, transb
       real(c_double), value :: alpha, beta
       type(c_ptr), value :: a, b, c
     end subroutine samo_dgemm_c
  end interface


contains

  function samo_dmalloc(device, nrows, ncols) result(r)
    implicit none
    integer :: device
    integer :: nrows, ncols
    type(c_ptr) :: cptr
    double precision, pointer :: a(:,:)
    type(samo_dmatrix) :: r
    r%p = samo_dmalloc_c(device, nrows*1_c_int64_t, ncols*1_c_int64_t)
    cptr = samo_dget_pointer_c(r%p)
    call c_f_pointer(cptr, r%m, (/ nrows, ncols /))
  end function samo_dmalloc

  subroutine samo_dfree(p)
    implicit none
    type(samo_dmatrix), intent(inout) :: p
    call samo_dfree_c(p%p)
    NULLIFY(p%m)
  end subroutine samo_dfree


  subroutine samo_dgemm(transa, transb, alpha, a, b, beta, c) 
    implicit none
    character, intent(in) :: transa, transb
    double precision, intent(in) :: alpha, beta
    type(samo_dmatrix), intent(in) :: a, b
    type(samo_dmatrix), intent(out) :: c
    call samo_dgemm_c(transa, transb, alpha, a%p, b%p, beta, c%p)
  end subroutine samo_dgemm

end module samo
