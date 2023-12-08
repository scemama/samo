use ::std::os::raw::c_void;

#[derive(Debug,Clone)]
pub struct Stream {
}

impl Stream {

    pub fn new() -> Self {
        Stream {}
    }

    pub fn as_raw_mut_ptr(&self) -> *mut c_void {
        0 as *mut c_void
    }

    pub fn as_raw_ptr(&self) -> *const c_void {
        0 as *const c_void
    }

    pub fn synchronize(&self) {
    }

}


