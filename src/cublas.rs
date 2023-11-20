pub mod cublas;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cublasContext {
    _unused: [u8; 0],
}
pub type cublasHandle_t = *mut cublasContext;

pub type cublasStatus_t = ::std::os::raw::c_uint;
pub const cublasStatus_t_CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;
pub const cublasStatus_t_CUBLAS_STATUS_NOT_INITIALIZED: cublasStatus_t = 1;
pub const cublasStatus_t_CUBLAS_STATUS_ALLOC_FAILED: cublasStatus_t = 3;
pub const cublasStatus_t_CUBLAS_STATUS_INVALID_VALUE: cublasStatus_t = 7;
pub const cublasStatus_t_CUBLAS_STATUS_ARCH_MISMATCH: cublasStatus_t = 8;
pub const cublasStatus_t_CUBLAS_STATUS_MAPPING_ERROR: cublasStatus_t = 11;
pub const cublasStatus_t_CUBLAS_STATUS_EXECUTION_FAILED: cublasStatus_t = 13;
pub const cublasStatus_t_CUBLAS_STATUS_INTERNAL_ERROR: cublasStatus_t = 14;
pub const cublasStatus_t_CUBLAS_STATUS_NOT_SUPPORTED: cublasStatus_t = 15;
pub const cublasStatus_t_CUBLAS_STATUS_LICENSE_ERROR: cublasStatus_t = 16;

pub type cublasOperation_t = ::std::os::raw::c_uint;
pub const cublasOperation_t_CUBLAS_OP_N: cublasOperation_t = 0;
pub const cublasOperation_t_CUBLAS_OP_T: cublasOperation_t = 1;
pub const cublasOperation_t_CUBLAS_OP_C: cublasOperation_t = 2;
pub const cublasOperation_t_CUBLAS_OP_HERMITAN: cublasOperation_t = 2;
pub const cublasOperation_t_CUBLAS_OP_CONJG: cublasOperation_t = 3;

extern "C" {
    // Context
    pub fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t;
    pub fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t;

    // Memory operations
    pub fn cublasSetMatrix(
        rows: ::std::os::raw::c_int,
        cols: ::std::os::raw::c_int,
        elemSize: ::std::os::raw::c_int,
        A: *const ::std::os::raw::c_void,
        lda: ::std::os::raw::c_int,
        B: *mut ::std::os::raw::c_void,
        ldb: ::std::os::raw::c_int,
    ) -> cublasStatus_t;

    pub fn cublasScopy_v2_64(
        handle: cublasHandle_t,
        n: i64,
        x: *const f32,
        incx: i64,
        y: *mut f32,
        incy: i64,
    ) -> cublasStatus_t;

    pub fn cublasGetMatrix_64(
        rows: i64,
        cols: i64,
        elemSize: i64,
        A: *const ::std::os::raw::c_void,
        lda: i64,
        B: *mut ::std::os::raw::c_void,
        ldb: i64,
    ) -> cublasStatus_t;

    // Stream operations
    pub fn cublasSetStream_v2(handle: cublasHandle_t, streamId: cudaStream_t) -> cublasStatus_t;
    pub fn cublasGetStream_v2(
        handle: cublasHandle_t,
        streamId: *mut cudaStream_t,
    ) -> cublasStatus_t;

    // BLAS
    pub fn cublasSgemv_v2_64(
        handle: cublasHandle_t,
        trans: cublasOperation_t,
        m: i64,
        n: i64,
        alpha: *const f32,
        A: *const f32,
        lda: i64,
        x: *const f32,
        incx: i64,
        beta: *const f32,
        y: *mut f32,
        incy: i64,
    ) -> cublasStatus_t;

    pub fn cublasDgemv_v2_64(
        handle: cublasHandle_t,
        trans: cublasOperation_t,
        m: i64,
        n: i64,
        alpha: *const f64,
        A: *const f64,
        lda: i64,
        x: *const f64,
        incx: i64,
        beta: *const f64,
        y: *mut f64,
        incy: i64,
    ) -> cublasStatus_t;

    pub fn cublasSgemm_v2_64(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i64,
        n: i64,
        k: i64,
        alpha: *const f32,
        A: *const f32,
        lda: i64,
        B: *const f32,
        ldb: i64,
        beta: *const f32,
        C: *mut f32,
        ldc: i64,
    ) -> cublasStatus_t;

    pub fn cublasDgemm_v2_64(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i64,
        n: i64,
        k: i64,
        alpha: *const f64,
        A: *const f64,
        lda: i64,
        B: *const f64,
        ldb: i64,
        beta: *const f64,
        C: *mut f64,
        ldc: i64,
    ) -> cublasStatus_t;

    pub fn cublasSgeam_64(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i64,
        n: i64,
        alpha: *const f32,
        A: *const f32,
        lda: i64,
        beta: *const f32,
        B: *const f32,
        ldb: i64,
        C: *mut f32,
        ldc: i64,
    ) -> cublasStatus_t;

    pub fn cublasDgeam_64(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i64,
        n: i64,
        alpha: *const f64,
        A: *const f64,
        lda: i64,
        beta: *const f64,
        B: *const f64,
        ldb: i64,
        C: *mut f64,
        ldc: i64,
    ) -> cublasStatus_t;

}


