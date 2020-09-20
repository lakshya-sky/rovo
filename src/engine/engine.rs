use super::task::*;
use crate::core::AutoGradMode;
use crate::ops::*;
use crate::util;
use crate::{ops::Node, tensor::*};
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

pub struct Engine {
    local_ready_queue: Rc<RefCell<ReadyQueue>>,
}

impl Engine {
    pub fn get_default_engine() -> Engine {
        Self {
            local_ready_queue: Rc::new(RefCell::new(ReadyQueue::new())),
        }
    }

    pub fn compute_dependencies(root: *const Node, graph_task: &mut GraphTask) {
        let mut seen: HashSet<*const Node> = HashSet::new();
        let mut queue: Vec<*const Node> = vec![root];
        let dependencies = &mut graph_task.dependencies.borrow_mut();
        loop {
            // eprintln!("dependencies: {:?}", dependencies);
            if queue.is_empty() {
                break;
            }
            let _fn = queue.pop();
            let edge = unsafe { &*_fn.unwrap() };
            if let Some(next_edges) = edge.next_edges() {
                for t in next_edges {
                    if let Some(next_ptr) = t.function.as_ref() {
                        let l = next_ptr.as_ptr();
                        *(dependencies.entry(l).or_insert(0)) += 1;
                        let was_inserted = seen.insert(l);
                        if was_inserted {
                            queue.push(l);
                        }
                    }
                }
            }
        }
    }

    pub fn evaluate_outputs(edges: Option<&Vec<Edge>>, mut grads: VariableList) -> VariableList {
        if let Some(edges) = edges {
            if edges.len() != grads.len() {
                panic!(format!(
                    "Invalid number of gradients - expected {}, but got {}",
                    edges.len(),
                    grads.len()
                ))
            }
            let mut new_grads = Vec::with_capacity(grads.len());
            let mut i = 0;
            let grad_len = grads.len();
            // println!("Grads Length: {}", grads.len());
            loop {
                if i >= grad_len {
                    break;
                }
                // println!("Index : {}", i);
                let edge = edges.get(i).unwrap();
                let function = edge.function().unwrap().borrow();
                let metadata = function.input_metadata(edge.input_nr);
                // remove shrinks vector that's why can't use i so use 0 to always get first element.
                let grad = grads.remove(0);
                if grad.sizes() != metadata.shape() {
                    if !util::is_expandable_to(metadata.shape(), grad.sizes()) {
                        panic!(format!("invalid gradient at index {} - got {:?}, but expected shape comapatible with {:?}", i, grad.sizes(), metadata.shape()));
                    }
                    new_grads.push(util::sum_to(grad, metadata.shape()))
                } else {
                    new_grads.push(grad);
                }
                i += 1;
            }
            new_grads
        } else {
            grads
        }
    }

    pub fn call_function(func: *mut Node, inputs: InputBuffer) -> VariableList {
        let inputs = InputBuffer::variables(inputs);
        let fn_ = unsafe { &mut *func };
        // eprintln!("Calling {:?}", fn_._impl);
        let outputs = fn_.call(inputs);
        Self::evaluate_outputs(fn_.next_edges(), outputs)
    }

    pub fn evaluate_function(
        &mut self,
        graph_task: Rc<RefCell<GraphTask>>,
        func: Rc<RefCell<Node>>,
        inputs: InputBuffer,
    ) {
        let _fnc = func.as_ptr();
        let outputs = Self::call_function(func.as_ptr(), inputs);
        let fn_ = func.borrow_mut();
        let num_outputs = outputs.len();
        let mut i = 0usize;
        let task = graph_task.borrow();
        let mut dependencies = task.dependencies.borrow_mut();
        loop {
            if i >= num_outputs {
                break;
            }
            let output = outputs.get(i).unwrap().clone();
            let next = fn_.next_edge(i);
            if next.is_none() {
                continue;
            }
            let next = next.unwrap();
            let mut is_ready = false;
            let t = next.function.as_ref().unwrap().as_ptr() as *const Node;
            let it = dependencies.get_mut(&t);
            if it.is_none() {
                panic!()
            } else {
                let count = it.unwrap();
                *count -= 1;
                if *count == 0 {
                    let _ = dependencies.remove(&t);
                    is_ready = true;
                }
            }

            let mut queue = task.ready_queue.borrow_mut();
            let mut not_ready = task.not_ready_queue.borrow_mut();

            if let Some(input_buffer) = not_ready.get_mut(&t) {
                input_buffer.add(next.input_nr, output);
                if is_ready {
                    queue.push(NodeTask::new(
                        Rc::downgrade(&graph_task.clone()),
                        next.function.as_ref().unwrap().clone(),
                        not_ready.remove(&t).unwrap(),
                    ));
                }
            } else {
                let mut input_buffer = InputBuffer::new_with_size(unsafe { &*t }.num_inputs());
                input_buffer.add(next.input_nr, output);
                if is_ready {
                    {
                        queue.push(NodeTask::new(
                            Rc::downgrade(&graph_task.clone()),
                            next.function.as_ref().unwrap().clone(),
                            input_buffer,
                        ));
                    }
                // let outstanding_task = task.outstanding_tasks.as_ptr();
                // unsafe { *outstanding_task = *outstanding_task + 1 };
                } else {
                    not_ready.insert(t, input_buffer);
                }
            }
            i += 1;
        }
    }

    pub fn thread_main(&mut self, graph_task: &Rc<RefCell<GraphTask>>) {
        let graph_task = graph_task.borrow();
        loop {
            let local_graph_task;
            {
                let task = self.local_ready_queue.borrow_mut().pop();
                if let Some(graph_task) = task.base_.upgrade() {
                    local_graph_task = graph_task;
                } else {
                    continue;
                }
                let _autograd_mode =
                    AutoGradMode::new(unsafe { &*local_graph_task.as_ptr() }.grad_mode);
                self.evaluate_function(local_graph_task.clone(), task.fn_, task.inputs_);
            }

            let outstanding_task = graph_task.outstanding_tasks.as_ptr();
            unsafe { *outstanding_task = *outstanding_task - 1 };

            if graph_task.completed() {
                break;
            }
        }
    }

    pub fn execute_with_graph_task(
        &mut self,
        task: &Rc<RefCell<GraphTask>>,
        root: Rc<RefCell<Node>>,
    ) {
        task.borrow().ready_queue.borrow_mut().push(NodeTask::new(
            Rc::downgrade(&task.clone()),
            root,
            InputBuffer::new_with_size(0),
        ));
        // let outstanding_task = task.borrow().outstanding_tasks.as_ptr();
        // unsafe { *outstanding_task = *outstanding_task + 1 };
        self.thread_main(task);
    }

    pub fn execute(
        &mut self,
        roots: EdgeList,
        inputs: VariableList,
        create_graph: bool,
        _output_edges: &mut EdgeList,
    ) {
        // println!("Inpots to Graph Root: {:?}", inputs);
        let graph_root = Node::new(GraphRoot::new(roots, inputs));
        let mut task = GraphTask::new(create_graph, 0, self.local_ready_queue.clone());
        Self::compute_dependencies(&graph_root, &mut task);
        let task = Rc::new(RefCell::new(task));
        self.execute_with_graph_task(&task, Rc::new(RefCell::new(graph_root)))
    }
}
