use crate::ops::*;
use crate::tensor::*;
use std::cell::RefCell;
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
        // let graph_task = Weak::upgrade(&task.base_).unwrap();
        // graph_task.borrow_mut().outstanding_tasks += 1;
        self.heap.push_back(task);
    }

    pub fn pop(&mut self) -> NodeTask {
        self.heap.pop_front().unwrap()
    }
}

pub struct GraphTask {
    pub dependencies: HashMap<*const Node, usize>,
    pub depth: usize,
    pub ready_queue: Rc<RefCell<ReadyQueue>>,
    pub outstanding_tasks: u32,
    pub grad_mode: bool
}

// Todo: Use ReadyQueue instead of VecDeque for push logic which increaments outstanding_task.
impl GraphTask {
    pub fn new(grad_mode: bool, depth: usize, ready_queue: Rc<RefCell<ReadyQueue>>) -> Self {
        let t: HashMap<*const Node, usize> = HashMap::new();
        GraphTask {
            depth,
            dependencies: t,
            ready_queue,
            outstanding_tasks: 0,
            grad_mode 
        }
    }

    pub fn completed(&self) -> bool {
        self.outstanding_tasks == 0
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

    pub fn add(&mut self, pos: usize, var: Tensor) {
        assert!(pos < self.buffer.capacity());
        self.buffer.insert(pos, var);
    }

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
