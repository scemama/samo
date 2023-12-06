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

  real*8, allocatable :: a(:,:), b(:,:), c(:,:), c_ref(:,:)
  type(c_ptr) :: a_tiled
  type(c_ptr) :: b_tiled
  type(c_ptr) :: c_tiled
  type(c_ptr) :: c2_tiled

  integer :: m, n, k, i, j
  type(c_ptr) :: h(10)

  double precision :: t0, t1

  m = 10100
  n = 6020
  k = 60300

  allocate(a(m,k))
  allocate(b(k,n))
  allocate(c(m,n))
  allocate(c_ref(m,n))

  !$OMP PARALLEL DO PRIVATE(i,j)
  do j=1, k
    do i=1, m
      a(i,j) = dble(i) + 10.d0*dble(j)
    enddo
  enddo

  !$OMP PARALLEL DO PRIVATE(i,j)
  do j=1, n
    do i=1, k
      b(i,j) = -dble(i) + 7.d0*dble(j)
    enddo
  enddo

  !$OMP PARALLEL DO PRIVATE(i,j)
  do j=1, n
    do i=1, m
      c(i,j) = 0.d0
      c_ref(i,j) = 0.d0
    enddo
  enddo


  call wall_time(t0)
  call dgemm('N','N', m, n, k, 0.5d0, a, m, b, k, 0.d0, c_ref, m)
  call wall_time(t1)
  print *, 'Time for DGEMM: ', t1-t0

  a_tiled = samo_dtile(a, m*1_8, k*1_8, m*1_8)
  b_tiled = samo_dtile(b, k*1_8, n*1_8, k*1_8)
  c_tiled = samo_dtile(c, m*1_8, n*1_8, m*1_8)


  call wall_time(t0)
  call samo_dgemm_tiled('N','N', 0.5d0, a_tiled, b_tiled, 0.d0, c_tiled)
  call wall_time(t1)
  print *, 'Time for Tiled DGEMM: ', t1-t0

  call samo_dfree(a_tiled)
  call samo_dfree(b_tiled)

  call samo_duntile(c_tiled, c, m*1_8)

  call samo_dfree(c_tiled)

  do j=1,n
    do i=1,m
      if (c(i,j) /= c_ref(i,j)) then
        print *, i, j, c(i,j), c_ref(i,j)
        stop
      endif
    enddo
  enddo


  a_tiled = samo_dtile_gpu(a, m*1_8, k*1_8, m*1_8)
  b_tiled = samo_dtile_gpu(b, k*1_8, n*1_8, k*1_8)
  c_tiled = samo_dtile_gpu(c, m*1_8, n*1_8, m*1_8)


  call samo_set_device(0)
  call wall_time(t0)
  print *, 'before'
  call samo_dgemm_tiled_gpu('N','N', 0.5d0, a_tiled, b_tiled, 0.d0, c_tiled)
  print *, 'after'
  call wall_time(t1)
  print *, 'Time for Tiled DGEMM: ', t1-t0

  call wall_time(t0)
  print *, 'before'
  call samo_dgemm_tiled_gpu('N','N', 0.5d0, a_tiled, b_tiled, 0.d0, c_tiled)
  print *, 'after'
  call wall_time(t1)
  print *, 'Time for Tiled DGEMM: ', t1-t0

  call samo_set_device(1)
  call wall_time(t0)
  print *, 'before'
  call samo_dgemm_tiled_gpu('N','N', 0.5d0, a_tiled, b_tiled, 0.d0, c_tiled)
  print *, 'after'
  call wall_time(t1)
  print *, 'Time for Tiled DGEMM: ', t1-t0

  call wall_time(t0)
  print *, 'before'
  h(1) = samo_dgemm_tiled_gpu_async('N','N', 0.5d0, a_tiled, b_tiled, 0.d0, c_tiled)
  print *, 'after'
  call samo_await(h(1))
  call wall_time(t1)
  print *, 'Time for Tiled DGEMM: ', t1-t0
  h(2) = samo_duntile_gpu_async(c_tiled, c, m*1_8)

  call samo_dfree_gpu(a_tiled)
  call samo_dfree_gpu(b_tiled)

  call samo_await(h(2))
  call samo_dfree_gpu(c_tiled)

  do j=1,n
    do i=1,m
      if (c(i,j) /= c_ref(i,j)) then
        print *, i, j, c(i,j), c_ref(i,j)
        stop
      endif
    enddo
  enddo

  call samo_set_device(0)
  a_tiled = samo_dtile_gpu(a, m*1_8, k*1_8, m*1_8)
  b_tiled = samo_dtile_gpu(b, k*1_8, n*1_8, k*1_8)
  c_tiled = samo_dtile_gpu(c, m*1_8, n*1_8, m*1_8)

  call samo_set_device(1)
  c2_tiled = samo_dtile_gpu(c, m*1_8, n*1_8, m*1_8)

  call wall_time(t0)
  call samo_set_device(1)
  h(1) = samo_dgemm_tiled_gpu_async('N','N', 0.5d0, a_tiled, b_tiled, 0.d0, c_tiled)
  call samo_set_device(2)
  h(2) = samo_dgemm_tiled_gpu_async('N','N', 0.5d0, a_tiled, b_tiled, 0.d0, c2_tiled)
  call samo_await(h(1))
  call samo_await(h(2))
  call wall_time(t1)
  print *, 'Time for 2 Tiled DGEMM: ', t1-t0
  call samo_duntile_gpu(c2_tiled, c, m*1_8)

  do j=1,n
    do i=1,m
      if (c(i,j) /= c_ref(i,j)) then
        print *, i, j, c(i,j), c_ref(i,j)
        stop
      endif
    enddo
  enddo

  deallocate(a, b)
  deallocate(c, c_ref)
end program

