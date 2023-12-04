/*
macro_rules! def_samo_config {
    ($cfg:literal) => {
        #[cfg(feature = $cfg)]
        const SAMO_CONFIG: &str = $cfg;
    };
}

def_samo_config!("netlib");
def_samo_config!("intel-mkl");
def_samo_config!("blis");
def_samo_config!("openblas");
def_samo_config!("accelerate");

// Default value
#[cfg(all(
    not(feature = "netlib"),
    not(feature = "intel-mkl"),
    not(feature = "blis"),
    not(feature = "openblas"),
    not(feature = "accelerate"),
))]
const SAMO_CONFIG: &str = "intel-mkl";

    let cfg = Config::from_str(SAMO_CONFIG).unwrap();
*/

fn main() {
    if cfg!(feature = "cublas") {
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cublasLt");
    }
}
