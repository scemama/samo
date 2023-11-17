#include "samo_f.F90"

subroutine wall_time(t)
  implicit none
  double precision, intent(out)  :: t
  integer*8                        :: c
  integer*8, save                  :: rate = 0
  if (rate == 0) then
    CALL SYSTEM_CLOCK(count_rate=rate)
  endif
  CALL SYSTEM_CLOCK(count=c)
  t = dble(c)/dble(rate)
end


program test
  use samo

  real*4, allocatable :: a(:,:), b(:,:), c(:,:), c_ref(:,:)
  type(c_ptr) :: a_tiled
  type(c_ptr) :: b_tiled
  type(c_ptr) :: c_tiled

  integer :: m, n, k, i, j

  double precision :: t0, t1

  m = 10100
  n = 20200
  k = 6030

  allocate(a(m,k))
  allocate(b(k,n))
  allocate(c(m,n))
  allocate(c_ref(m,n))

  do j=1, k
    do i=1, m
      a(i,j) = dble(i) + 10.d0*dble(j)
    enddo
  enddo

  do j=1, n
    do i=1, k
      b(i,j) = -dble(i) + 7.d0*dble(j)
    enddo
  enddo

  c(:,:) = 0.d0
  c_ref(:,:) = 0.d0


  call wall_time(t0)
  call sgemm('N','N', m, n, k, 1.0, a, m, b, k, 0.0, c_ref, m)
  call wall_time(t1)
  print *, 'Time for DGEMM: ', t1-t0

  a_tiled = samo_stile(a, m*1_8, k*1_8, m*1_8)
  print *, a(1,1)
  print *, a(m,k)
  b_tiled = samo_stile(b, k*1_8, n*1_8, k*1_8)
  print *, b(1,1)
  print *, b(k,n)
  c_tiled = samo_stile(c, m*1_8, n*1_8, m*1_8)
  print *, c(1,1)
  print *, c(m,n)

  deallocate(a, b)

  call wall_time(t0)
  call samo_sgemm_tiled('N','N', 1.0, a_tiled, b_tiled, 0.0, c_tiled)
  call wall_time(t1)
  print *, 'Time for Tiled DGEMM: ', t1-t0

  call samo_sfree(a_tiled)
  call samo_sfree(b_tiled)

  call samo_suntile(c_tiled, c, m*1_8)

  call samo_sfree(c_tiled)

  do j=1,n
    do i=1,m
      if (c(i,j) /= c_ref(i,j)) then
        print *, i, j, c(i,j), c_ref(i,j)
        stop
      endif
    enddo
  enddo

  call wall_time(t0)
  call sgemm('N','N', m, n, k, 1.0, a, m, b, k, 0.0, c_ref, m)
  call wall_time(t1)
  print *, 'Time for DGEMM: ', t1-t0

  deallocate(c, c_ref)
end program

