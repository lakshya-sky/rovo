// use std::collections::VecDeque;

// pub trait TaskThreadPoolBase {
//     fn defaultNumThreads() -> usize {
//         // Todo: Pytorch c++ uses hardware_concurrency
//         let num_threads = 4;
//         num_threads
//     }

//     fn run<F>(func: F)
//     where
//         F: FnMut() -> ();

//     fn size(&self) -> usize;

//     fn numAvailable(&self) -> usize;

//     fn inThreadPool(&self) -> bool;
// }

// mod thread_pool {
//     use std::collections::VecDeque;

//     struct TaskElement {
//         run_with_id: bool,
//         no_id: Option<Box<dyn FnMut() -> ()>>,
//         with_id: Option<Box<dyn FnMut(usize) -> ()>>,
//     }
//     impl TaskElement {
//         pub fn new_no_id<F>(f: F) -> Self
//         where
//             F: FnMut() -> () + 'static,
//         {
//             Self {
//                 run_with_id: false,
//                 no_id: Some(Box::new(f)),
//                 with_id: None,
//             }
//         }
//         pub fn new_with_id<F>(f: F) -> Self
//         where
//             F: FnMut(usize) -> () + 'static,
//         {
//             Self {
//                 run_with_id: false,
//                 no_id: None,
//                 with_id: Some(Box::new(f)),
//             }
//         }
//     }
//     struct ThreadPool {
//         tasks: VecDeque<TaskElement>,
//         threads: Vec<thread
//     }
// }
