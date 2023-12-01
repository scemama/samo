# SAMO
Synchronous/Asynchronous Matrix Operations

## Installation

```bash
cargo build --features <feature>
```

where `<feature>` is the BLAS feature:
* `accelerate`, which is the one in the Accelerate framework (macOS only),
* `blis`, which is the one in BLIS,
* `intel-mkl`, which is the one in Intel MKL,
* `netlib`, which is the reference one by Netlib,
* `openblas`, which is the one in OpenBLAS, and

## Performance

To enable maximum performance,

```bash
export RUSTFLAGS="-Ctarget-cpu=native"
```

Recommended features for compiling the library:
* x86: `intel-mkl`
* Others: `openblas` or `blis`

