use crate::cublas;
use crate::cuda::Device;
use std::thread;
use std::sync::mpsc;
use std::time::Duration;
use crate::Matrix;
use std::sync::Mutex;

pub enum TransAB { NN, NT, TN, TT }

pub enum Task<T: Sync + Send> {
    Free(*mut T),
    Reshape( *mut Matrix<T>, usize, usize ),
    Gemm(TransAB, T, *const Matrix<T>, *const Matrix<T>, T, *mut Matrix<T> ),
    Geam(TransAB, T, *const Matrix<T>, T, *const Matrix<T>, *mut Matrix<T> ),
    Sleep(u64),
    Terminate,
}

unsafe impl<T: Sync+Send> Send for Task<T> {}
unsafe impl<T: Sync+Send> Sync for Task<T> {}

pub struct Stream<T: Sync + Send> {
    thread: thread::JoinHandle<()>,
    tx: mpsc::Sender<Task<T>>,
}


macro_rules! make_stream { ($s:ty) => {

impl Stream<$s> {

    pub fn run(handle: Option<&cublas::Context>, task: Task<$s>) {
        match task {
            Task::Free(a) => {
                drop( unsafe { Box::from_raw(a) } );
            },

            Task::Reshape(a, nrows, ncols) => {
                unsafe {
                (*a).reshape(nrows, ncols);
                }
            },

            Task::Gemm( TransAB::NN, alpha, a, b, beta, c ) => {
                unsafe{
                    Matrix::<$s>::gemm_mut(handle, alpha, &*a, &*b, beta, &mut *c);
                }
            },
            Task::Gemm( TransAB::NT, alpha, a, b, beta, c ) => {
                unsafe{
                    let b = &(*b).t();
                    Matrix::<$s>::gemm_mut(handle, alpha, &*a, &*b, beta, &mut *c);
                }
            },
            Task::Gemm( TransAB::TN, alpha, a, b, beta, c ) => {
                unsafe{
                    let a = &(*a).t();
                    Matrix::<$s>::gemm_mut(handle, alpha, &*a, &*b, beta, &mut *c);
                }
            },
            Task::Gemm( TransAB::TT, alpha, a, b, beta, c ) => {
                unsafe{
                    let a = &(*a).t();
                    let b = &(*b).t();
                    Matrix::<$s>::gemm_mut(handle, alpha, &*a, &*b, beta, &mut *c);
                }
            },

            Task::Geam( TransAB::NN, alpha, a, beta, b, c ) => {
                unsafe{
                    Matrix::<$s>::geam_mut(handle, alpha, &*a, beta, &*b, &mut *c);
                }
            },
            Task::Geam( TransAB::NT, alpha, a, beta, b, c ) => {
                unsafe{
                    let b = &(*b).t();
                    Matrix::<$s>::geam_mut(handle, alpha, &*a, beta, &*b, &mut *c);
                }
            },
            Task::Geam( TransAB::TN, alpha, a, beta, b, c ) => {
                unsafe{
                    let a = &(*a).t();
                    Matrix::<$s>::geam_mut(handle, alpha, &*a, beta, &*b, &mut *c);
                }
            },
            Task::Geam( TransAB::TT, alpha, a, beta, b, c ) => {
                unsafe{
                    let a = &(*a).t();
                    let b = &(*b).t();
                    Matrix::<$s>::geam_mut(handle, alpha, &*a, beta, &*b, &mut *c);
                }
            },

            Task::Sleep(d) => {
                thread::sleep(Duration::from_millis(d));
            },

            _ => unreachable!(),
        }
    }

    pub fn new(device: Device) -> Self {
        let (tx, rx): (mpsc::Sender<Task<$s>>, mpsc::Receiver<Task<$s>>) = mpsc::channel();
        let t = thread::spawn(move || {
            let ctx = cublas::Context::new().unwrap();
            let handle = match device {
                    Device::CPU => None,
                    Device::GPU(_) => Some(&ctx),
            };
            while true {
                let task = rx.recv().unwrap();
                match task {
                    Task::Terminate => {
                        break;
                    },
                    task => Self::run(handle, task),
                }
            }
        });
        Self { thread: t, tx }
    }


    pub fn wait(self) {
        self.tx.send(Task::Terminate).unwrap();
        self.thread.join().unwrap();
    }

    pub fn push(&self, t: Task<$s>) {
        self.tx.send(t).unwrap();
    }
}

}}
make_stream!(f64);
make_stream!(f32);


#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Instant, Duration};

    #[test]
    fn single_stream() {
        let s = Stream::<f64>::new(Device::CPU);
        let time = Instant::now();
        s.push(Task::Sleep(10));
        let duration = time.elapsed();
        println!("Push task 1: {:?}", duration);
        s.push(Task::Sleep(10));
        let duration = time.elapsed();
        println!("Push task 2: {:?}", duration);
        s.wait();
        let duration = time.elapsed();
        println!("wait: {:?}", duration);
        assert!(duration > Duration::from_millis(19));
    }

    #[test]
    fn multiple_stream() {
        let s1 = Stream::<f64>::new(Device::CPU);
        let s2 = Stream::<f64>::new(Device::CPU);
        let s3 = Stream::<f64>::new(Device::CPU);
        let time = Instant::now();
        s1.push(Task::Sleep(10));
        s2.push(Task::Sleep(10));
        s3.push(Task::Sleep(10));
        let duration = time.elapsed();
        println!("Push task 1: {:?}", duration);
        s1.push(Task::Sleep(10));
        s2.push(Task::Sleep(10));
        s3.push(Task::Sleep(10));
        let duration = time.elapsed();
        println!("Push task 2: {:?}", duration);
        s1.wait();
        let duration = time.elapsed();
        println!("wait: {:?}", duration);
        s2.wait();
        let duration = time.elapsed();
        println!("wait: {:?}", duration);
        s3.wait();
        let duration = time.elapsed();
        println!("wait: {:?}", duration);
        assert!(duration > Duration::from_millis(19));
        assert!(duration < Duration::from_millis(25));
    }
}
