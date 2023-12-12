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

  real*8, allocatable :: c_ref(:,:)
  type(samo_dmatrix) :: a, b, c, c2

  integer :: m, n, k, i, j

  double precision :: t0, t1

  m = 1010
  n = 60200
  k = 60300
!
!  m = 101
!  n = 602
!  k = 604

  a = samo_dmalloc(0, m, k)
  b = samo_dmalloc(0, k, n)
  c = samo_dmalloc(0, m, n)
  allocate(c_ref(m,n))

  print *, 'Prepare A'
  !$OMP PARALLEL DO PRIVATE(i,j)
  do j=1, k
    do i=1, m
      a%m(i,j) = dble(i) + 10.d0*dble(j)
    enddo
  enddo

  print *, 'Prepare B'
  !$OMP PARALLEL DO PRIVATE(i,j)
  do j=1, n
    do i=1, k
      b%m(i,j) = -dble(i) + 7.d0*dble(j)
    enddo
  enddo

  print *, 'Prepare C'
  !$OMP PARALLEL DO PRIVATE(i,j)
  do j=1, n
    do i=1, m
      c%m(i,j) = 0.d0
      c_ref(i,j) = 0.d0
    enddo
  enddo

  print *, 'Preparation ok'

  call wall_time(t0)
  call samo_dgemm('N','N', 0.5d0, a, b, 0.d0, c)
  call wall_time(t1)
  print *, 'Time for SAMO DGEMM: ', t1-t0

  call wall_time(t0)
  call dgemm('N','N', m, n, k, 0.5d0, a%m(1,1), m, b%m(1,1), k, 0.d0, c_ref(1,1), m)
  call wall_time(t1)
  print *, 'Time for DGEMM: ', t1-t0

  print *, 'Compare'
  do j=1,n
    do i=1,m
      if (c%m(i,j) /= c_ref(i,j)) then
        print *, i, j, c%m(i,j), c_ref(i,j)
        stop
      endif
    enddo
  enddo
  print *, 'Done'

  call wall_time(t0)
  call samo_dgemm('N','N', 0.5d0, a, b, 0.d0, c)
  call wall_time(t1)
  print *, 'Time for SAMO DGEMM: ', t1-t0

  call wall_time(t0)
  call samo_dgemm('N','N', 0.5d0, a, b, 0.d0, c)
  call wall_time(t1)
  print *, 'Time for SAMO DGEMM: ', t1-t0


  call samo_dfree(a)
  call samo_dfree(b)
  call samo_dfree(c)
  deallocate(c_ref)
end program

