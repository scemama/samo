use std::thread;

pub struct Stream {
   last_task: Option< thread::JoinHandle<()> >,
}

impl Stream {

    pub fn new() -> Self {
        Self { last_task: None }
    }

    pub fn push(self, task: fn () -> () ) -> Self {
        Self { last_task: match self.last_task {
            None => {
                Some (thread::spawn(move || { task() }))
            },
            Some(prev_task) => {
                Some (thread::spawn(move || {
                    prev_task.join().unwrap();
                    task()
                }))
            },
        } }
    }

    pub fn wait(self) -> Self {
        match self.last_task {
            Some(handle) => handle.join().unwrap(),
            _ => (),
        };
        Self { last_task: None }
    }

}



#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Instant, Duration};

    #[test]
    fn single_stream() {
        let mut s = Stream::new();
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
        let mut s1 = Stream::new();
        let mut s2 = Stream::new();
        let mut s3 = Stream::new();
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
