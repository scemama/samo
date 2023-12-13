use super::*;

#[derive(Debug, Clone)]
pub struct Context();

impl Context {

    pub fn new() -> Result<Self, CublasError> {
        Ok( Context() )
    }

    /*
    pub fn as_raw_ptr(&self) -> *const c_void {
        self.0.as_raw_ptr()
    }
    */
}


