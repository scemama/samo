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
     subroutine samo_dcopy(source, destination) bind(C)
       import
       implicit none
       type(c_ptr), value        :: source, destination
     end subroutine samo_dcopy
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
     function samo_dsubmatrix_c(a, row, col, nrows, ncols) result(ptr) bind(C, name='samo_dsubmatrix')
       import
       implicit none
       type(c_ptr), value  :: a
       integer(c_int64_t), value  :: row, col, nrows, ncols
       type(c_ptr)         :: ptr
     end function samo_dsubmatrix_c
  end interface

  interface
     function samo_dget_pointer(a) result(ptr) bind(C)
       import
       implicit none
       type(c_ptr), value  :: a
       type(c_ptr)         :: ptr
     end function samo_dget_pointer
  end interface

  interface
     subroutine samo_dfree_c(p) bind(C, name='samo_dfree')
       import
       implicit none
       type(c_ptr), value :: p
     end subroutine samo_dfree_c
  end interface

  interface
     subroutine samo_dreshape_c(a, nrows, ncols) bind(C, name='samo_dreshape')
       import
       implicit none
       type(c_ptr), intent(in), value :: a
       integer(c_int64_t), intent(in), value :: nrows, ncols
     end subroutine samo_dreshape_c
  end interface

  interface
     subroutine samo_dgemm_nn_c(alpha, a, b, beta, c) bind(C, name='samo_dgemm_nn')
       import
       implicit none
       real(c_double), value :: alpha, beta
       type(c_ptr), value :: a, b, c
     end subroutine samo_dgemm_nn_c
  end interface

  interface
     subroutine samo_dgemm_nt_c(alpha, a, b, beta, c) bind(C, name='samo_dgemm_nt')
       import
       implicit none
       real(c_double), value :: alpha, beta
       type(c_ptr), value :: a, b, c
     end subroutine samo_dgemm_nt_c
  end interface

  interface
     subroutine samo_dgemm_tn_c(alpha, a, b, beta, c) bind(C, name='samo_dgemm_tn')
       import
       implicit none
       real(c_double), value :: alpha, beta
       type(c_ptr), value :: a, b, c
     end subroutine samo_dgemm_tn_c
  end interface

  interface
     subroutine samo_dgemm_tt_c(alpha, a, b, beta, c) bind(C, name='samo_dgemm_tt')
       import
       implicit none
       real(c_double), value :: alpha, beta
       type(c_ptr), value :: a, b, c
     end subroutine samo_dgemm_tt_c
  end interface

  interface
     subroutine samo_dgeam_nn_c(alpha, a, beta, b, c) bind(C, name='samo_dgeam_nn')
       import
       implicit none
       real(c_double), value :: alpha, beta
       type(c_ptr), value :: a, b, c
     end subroutine samo_dgeam_nn_c
  end interface

  interface
     subroutine samo_dgeam_nt_c(alpha, a, beta, b, c) bind(C, name='samo_dgeam_nt')
       import
       implicit none
       real(c_double), value :: alpha, beta
       type(c_ptr), value :: a, b, c
     end subroutine samo_dgeam_nt_c
  end interface

  interface
     subroutine samo_dgeam_tn_c(alpha, a, beta, b, c) bind(C, name='samo_dgeam_tn')
       import
       implicit none
       real(c_double), value :: alpha, beta
       type(c_ptr), value :: a, b, c
     end subroutine samo_dgeam_tn_c
  end interface

  interface
     subroutine samo_dgeam_tt_c(alpha, a, beta, b, c) bind(C, name='samo_dgeam_tt')
       import
       implicit none
       real(c_double), value :: alpha, beta
       type(c_ptr), value :: a, b, c
     end subroutine samo_dgeam_tt_c
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
    cptr = samo_dget_pointer(r%p)
    call c_f_pointer(cptr, r%m, (/ nrows, ncols /))
  end function samo_dmalloc

  function samo_dsubmatrix(a, row, col, nrows, ncols) result(r)
    implicit none
    type(samo_dmatrix), intent(in) :: a
    integer, intent(in) :: row, col, nrows, ncols
    type(samo_dmatrix) :: r
    type(c_ptr) :: cptr
    r%p = samo_dsubmatrix_c(a%p, row*1_c_int64_t-1_c_int64_t, col*1_c_int64_t-1_c_int64_t, nrows*1_c_int64_t, ncols*1_c_int64_t)
    cptr = samo_dget_pointer(r%p)
    call c_f_pointer(cptr, r%m, (/ size(a%m,1), ncols /))
  end function samo_dsubmatrix

  subroutine samo_dreshape(a, nrows, ncols)
     import
     implicit none
     type(samo_dmatrix), intent(in) :: a
     integer, intent(in) :: nrows, ncols
     type(c_ptr) :: cptr
     call samo_dreshape_c(a%p, nrows*1_c_int64_t, ncols*1_c_int64_t)
     cptr = samo_dget_pointer(a%p)
     call c_f_pointer(cptr, a%m, (/ nrows, ncols /))
  end subroutine samo_dreshape

  subroutine samo_dfree(p)
    implicit none
    type(samo_dmatrix), intent(inout) :: p
    call samo_dfree_c(p%p)
    NULLIFY(p%m)
  end subroutine samo_dfree


  subroutine samo_dgemm_nn(alpha, a, b, beta, c)
    implicit none
    double precision, intent(in) :: alpha, beta
    type(samo_dmatrix), intent(in) :: a, b
    type(samo_dmatrix), intent(out) :: c
    call samo_dgemm_nn_c(alpha, a%p, b%p, beta, c%p)
  end subroutine samo_dgemm_nn

  subroutine samo_dgemm_nt(alpha, a, b, beta, c)
    implicit none
    double precision, intent(in) :: alpha, beta
    type(samo_dmatrix), intent(in) :: a, b
    type(samo_dmatrix), intent(out) :: c
    call samo_dgemm_nt_c(alpha, a%p, b%p, beta, c%p)
  end subroutine samo_dgemm_nt

  subroutine samo_dgemm_tn(alpha, a, b, beta, c)
    implicit none
    double precision, intent(in) :: alpha, beta
    type(samo_dmatrix), intent(in) :: a, b
    type(samo_dmatrix), intent(out) :: c
    call samo_dgemm_tn_c(alpha, a%p, b%p, beta, c%p)
  end subroutine samo_dgemm_tn

  subroutine samo_dgemm_tt(alpha, a, b, beta, c)
    implicit none
    double precision, intent(in) :: alpha, beta
    type(samo_dmatrix), intent(in) :: a, b
    type(samo_dmatrix), intent(out) :: c
    call samo_dgemm_tt_c(alpha, a%p, b%p, beta, c%p)
  end subroutine samo_dgemm_tt


  subroutine samo_dgeam_nn(alpha, a, beta, b, c)
    implicit none
    double precision, intent(in) :: alpha, beta
    type(samo_dmatrix), intent(in) :: a, b
    type(samo_dmatrix), intent(out) :: c
    call samo_dgeam_nn_c(alpha, a%p, beta, b%p, c%p)
  end subroutine samo_dgeam_nn

  subroutine samo_dgeam_nt(alpha, a, beta, b, c)
    implicit none
    double precision, intent(in) :: alpha, beta
    type(samo_dmatrix), intent(in) :: a, b
    type(samo_dmatrix), intent(out) :: c
    call samo_dgeam_nt_c(alpha, a%p, beta, b%p, c%p)
  end subroutine samo_dgeam_nt

  subroutine samo_dgeam_tn(alpha, a, beta, b, c)
    implicit none
    double precision, intent(in) :: alpha, beta
    type(samo_dmatrix), intent(in) :: a, b
    type(samo_dmatrix), intent(out) :: c
    call samo_dgeam_tn_c(alpha, a%p, beta, b%p, c%p)
  end subroutine samo_dgeam_tn

  subroutine samo_dgeam_tt(alpha, a, beta, b, c)
    implicit none
    double precision, intent(in) :: alpha, beta
    type(samo_dmatrix), intent(in) :: a, b
    type(samo_dmatrix), intent(out) :: c
    call samo_dgeam_tt_c(alpha, a%p, beta, b%p, c%p)
  end subroutine samo_dgeam_tt

end module samo
