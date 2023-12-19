use crate::cublas;
use std::thread;

pub enum Stream {
   CPU(Option<thread::JoinHandle<()>>),
   GPU(Option<cublas::Context>),
}

impl Stream {

    pub fn new_cpu() -> Self {
        Stream::CPU(None)
    }

    pub fn new_gpu() -> Self {
        Stream::GPU(None)
    }

    pub fn push(self, task: fn () -> () ) -> Self {
        match self {
            Stream::CPU(None) => Stream::CPU( Some (thread::spawn(move || { task() })) ),
            Stream::GPU(None) => Stream::GPU( Some (cublas::Context::new().unwrap()) ),
            Stream::CPU(Some(prev_task)) => {
                Stream::CPU( Some (thread::spawn(move || {
                    prev_task.join().unwrap();
                    task()
                })) ) },
            Stream::GPU(Some(context)) => Stream::GPU(Some(context)),
        }
    }

    pub fn wait(self) -> Self {
        match self {
            Stream::CPU(None) => Stream::CPU(None),
            Stream::GPU(None) => Stream::GPU(None),
            Stream::CPU(Some(prev_task)) => {
                    prev_task.join().unwrap();
                    Stream::CPU(None) },
            Stream::GPU(Some(context)) => {
                drop(context);
                Stream::GPU(None)
            },
        }
    }

}



#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Instant, Duration};

    #[test]
    fn single_stream() {
        let mut s = Stream::new_cpu();
        let time = Instant::now();
        s = s.push( || { thread::sleep(Duration::from_millis(10)) } );
        let duration = time.elapsed();
        println!("Push task 1: {:?}", duration);
        s = s.push( || { thread::sleep(Duration::from_millis(10)) } );
        let duration = time.elapsed();
        println!("Push task 2: {:?}", duration);
        s = s.wait();
        let duration = time.elapsed();
        println!("wait: {:?}", duration);
        assert!(duration > Duration::from_millis(19));
    }

    #[test]
    fn multiple_stream() {
        let mut s1 = Stream::new_cpu();
        let mut s2 = Stream::new_cpu();
        let mut s3 = Stream::new_cpu();
        let time = Instant::now();
        s1 = s1.push( || { thread::sleep(Duration::from_millis(10)) } );
        s2 = s2.push( || { thread::sleep(Duration::from_millis(10)) } );
        s3 = s3.push( || { thread::sleep(Duration::from_millis(10)) } );
        let duration = time.elapsed();
        println!("Push task 1: {:?}", duration);
        s1 = s1.push( || { thread::sleep(Duration::from_millis(10)) } );
        s2 = s2.push( || { thread::sleep(Duration::from_millis(10)) } );
        s3 = s3.push( || { thread::sleep(Duration::from_millis(10)) } );
        let duration = time.elapsed();
        println!("Push task 2: {:?}", duration);
        s1 = s1.wait();
        let duration = time.elapsed();
        println!("wait: {:?}", duration);
        s2 = s2.wait();
        let duration = time.elapsed();
        println!("wait: {:?}", duration);
        s3 = s3.wait();
        let duration = time.elapsed();
        println!("wait: {:?}", duration);
        assert!(duration > Duration::from_millis(19));
        assert!(duration < Duration::from_millis(25));
    }
}
