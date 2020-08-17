use crate::ops::*;
use crate::tensor::*;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::rc::{Rc, Weak};

pub struct ReadyQueue {
    pub heap: VecDeque<NodeTask>,
}

impl ReadyQueue {
    pub fn new() -> Self {
        Self {
            heap: VecDeque::new(),
        }
    }

    pub fn push(&mut self, task: NodeTask) {
        let graph_task = Weak::upgrade(&task.base_).unwrap();
        let outstanding_task = graph_task.borrow().outstanding_tasks.as_ptr();
        unsafe { *outstanding_task = *outstanding_task + 1 };
        self.heap.push_back(task);
    }

    pub fn pop(&mut self) -> NodeTask {
        self.heap.pop_front().unwrap()
    }
}

pub struct GraphTask {
    pub dependencies: RefCell<HashMap<*const Node, usize>>,
    pub depth: usize,
    pub ready_queue: Rc<RefCell<ReadyQueue>>,
    pub outstanding_tasks: Cell<u32>,
    pub grad_mode: bool,
    pub not_ready_queue: RefCell<HashMap<*const Node, InputBuffer>>,
}

// Todo: Use ReadyQueue instead of VecDeque for push logic which increaments outstanding_task.
impl GraphTask {
    pub fn new(grad_mode: bool, depth: usize, ready_queue: Rc<RefCell<ReadyQueue>>) -> Self {
        let t: HashMap<*const Node, usize> = HashMap::new();
        let not_ready_queue: HashMap<*const Node, InputBuffer> = HashMap::new();
        GraphTask {
            depth,
            dependencies: RefCell::new(t),
            ready_queue,
            outstanding_tasks: Cell::new(0),
            grad_mode,
            not_ready_queue: RefCell::new(not_ready_queue),
        }
    }

    pub fn completed(&self) -> bool {
        let t = self.outstanding_tasks.get();
        t == 0
    }
}

pub struct InputBuffer {
    pub buffer: VariableList,
}

impl InputBuffer {
    pub fn new_with_size(size: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(size),
        }
    }

    pub fn variables(other: Self) -> VariableList {
        other.buffer
    }

    pub fn add(&mut self, pos: usize, tensor: Tensor) {
        assert!(pos < self.buffer.capacity());
        match self.buffer.get(pos) {
            Some(_) => accumulate(&mut self.buffer, pos, tensor),
            None => self.buffer.insert(pos, tensor),
        }
    }
}

pub fn accumulate(buffer: &mut VariableList, pos: usize, tensor: Tensor) {
    let old_var = buffer.remove(pos);
    //Todo: use sparse tensor logic. currently performs basic accumulation
    buffer.insert(pos, old_var + tensor);
}

pub struct NodeTask {
    pub base_: Weak<RefCell<GraphTask>>,
    pub fn_: Rc<RefCell<Node>>,
    pub inputs_: InputBuffer,
}

impl NodeTask {
    pub fn new(
        base_: Weak<RefCell<GraphTask>>,
        fn_: Rc<RefCell<Node>>,
        inputs_: InputBuffer,
    ) -> Self {
        Self {
            base_,
            fn_,
            inputs_,
        }
    }
}
