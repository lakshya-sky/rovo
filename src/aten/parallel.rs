use std::{
    cell::RefCell,
    sync::{Arc, Condvar, Mutex},
};

use crossbeam::atomic::AtomicCell;
use rayon::{ThreadPool, ThreadPoolBuilder};

#[inline(always)]
pub fn divup(x: usize, y: usize) -> usize {
    return (x + y - 1) / y;
}

pub const GRAIN_SIZE: usize = 32768;
pub fn parallel_for<F>(begin: usize, end: usize, grain_size: usize, mut f: F)
where
    F: FnMut(usize, usize) + Send,
{
    if begin >= end {
        return;
    }
    if end - begin < grain_size || in_parallel_region() {
        f(begin, end);
        return;
    }
    _parallel_run(
        begin,
        end,
        grain_size,
        |start: usize, end: usize, _: usize| {
            f(start, end);
        },
    );
}

fn in_parallel_region() -> bool {
    true
}

fn init_num_threads() {}
#[inline(always)]
pub fn lazy_init_num_threads() {
    thread_local! {
    static init:RefCell<bool>= RefCell::new(false);
      };
    init.with(|i| {
        if !*i.borrow() {
            init_num_threads();
            *i.borrow_mut() = true;
        }
    });
}

fn calc_num_tasks_and_chunk_size(begin: usize, end: usize, grain_size: usize) -> (usize, usize) {
    if (end - begin) < grain_size {
        return (1, 0usize.max(end - begin));
    }
    // Choose number of tasks based on grain size and number of threads.
    let mut chunk_size = divup(end - begin, get_num_threads());
    // Make sure each task is at least grain_size size.
    chunk_size = grain_size.max(chunk_size);
    let num_tasks = divup(end - begin, chunk_size);
    return (num_tasks, chunk_size);
}

// Todo: state has additional err_ptr in pytorch, which stores pointer to
// exception in C++, but I couldn't figure out how to do it in rust. so I am going ahed without it.
struct internal {
    pub remaining: Mutex<usize>,
    pub cv: Condvar,
}

impl Default for internal {
    fn default() -> Self {
        Self {
            remaining: Mutex::new(0),
            cv: Condvar::new(),
        }
    }
}
impl internal {
    pub fn with_remaining(remaining_task: usize) -> Self {
        Self {
            remaining: Mutex::new(remaining_task),
            cv: Condvar::new(),
        }
    }
}

fn _parallel_run<F>(begin: usize, end: usize, grain_size: usize, mut f: F)
where
    F: FnMut(usize, usize, usize) + Send,
{
    lazy_init_num_threads();
    let (num_tasks, chunk_size) = calc_num_tasks_and_chunk_size(begin, end, grain_size);

    let state = Arc::new(internal::with_remaining(num_tasks));
    let task = |_: usize, task_id: usize| {
        let local_start = begin + task_id * chunk_size;
        if local_start < end {
            let local_end = end.min(chunk_size + local_start);

            // try {
            //           ParallelRegionGuard guard(task_id);
            f(local_start, local_end, task_id);
            //         } catch (...) {
            //           if (!state.err_flag.test_and_set()) {
            //             state.eptr = std::current_exception();
            //           }
            // }
        }
        {
            let lock = (&state).clone();
            let mut remaining = lock.remaining.lock().unwrap();
            *remaining = *remaining - 1;
            if *remaining == 0 {
                lock.cv.notify_one();
            }
        }
    };
    _run_with_pool(task, num_tasks);
    // Wait for all tasks to finish.
    {
        let mut remaining = state.remaining.lock().unwrap();
        if *remaining != 0 {
            remaining = state.cv.wait(remaining).unwrap();
        }
    }
    // if (state.eptr) {
    //     std::rethrow_exception(state.eptr);
    // }
}

// Number of threads set by the user
// NOT_SET -> positive value -> CONSUMED
// or
// NOT_SET -> CONSUMED
// Meaning:
//  - NOT_SET - pool not initialized, user value is not set
//  - SET with value - pool not initialized, user value set
//  - CONSUMED - pool is initialized
const NOT_SET: isize = -2;
const CONSUMED: isize = -1;

pub fn get_num_intraop_threads() -> &'static AtomicCell<isize> {
    static num_intraop_threads: once_cell::sync::OnceCell<AtomicCell<isize>> =
        once_cell::sync::OnceCell::new();
    num_intraop_threads.get_or_init(|| AtomicCell::new(NOT_SET))
}

fn intraop_default_num_threads() -> isize {
    //Todo: Implement this whole funcion
    let nthreads = 4;
    nthreads
}

pub fn _num_pool_threads(mut nthreads: isize) -> isize {
    if nthreads == NOT_SET {
        nthreads = intraop_default_num_threads();
    } else {
        assert!(nthreads > 0);
    }
    // minus one because of the master thread
    return nthreads - 1;
}

pub fn _get_intraop_pool() -> &'static ThreadPool {
    static pool: once_cell::sync::OnceCell<ThreadPool> = once_cell::sync::OnceCell::new();
    let p = pool.get_or_init(|| {
        let num_threads = get_num_intraop_threads().swap(CONSUMED);
        let num_threads = _num_pool_threads(num_threads) as usize;
        let p = ThreadPoolBuilder::new().num_threads(num_threads).build();
        p.unwrap()
    });
    p
}
fn _run_with_pool<F>(mut f: F, range: usize)
where
    F: FnMut(usize, usize) + Send,
{
    for i in 1..range {
        _get_intraop_pool().install(|| f(i, i));
    }
    f(0, 0);
}

fn get_num_threads() -> usize {
    // not initializing pool unnecessarily,
    // because pool cannot be resized after initialization
    let num_intraop_threads = get_num_intraop_threads();
    let nthreads: isize = num_intraop_threads.load();
    if nthreads > 0 {
        return nthreads as usize;
    } else if nthreads == NOT_SET {
        return intraop_default_num_threads() as usize;
    } else {
        debug_assert_eq!(nthreads, CONSUMED);
        return _get_intraop_pool().current_num_threads() + 1;
    }
}
