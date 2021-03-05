use super::task::*;
use crate::aten;
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

    // This function takes edges and inputs to those edges. if the edge is valid then there should be corresponding
    // grad input. It loops in reverse order so grads vec doesn't have shift other elements. Due to that new grads will be in reverse order.
    // while returning them, the vector is reversed.
    pub fn validate_outputs(edges: Option<&Vec<Edge>>, mut grads: VariableList) -> VariableList {
        if let Some(edges) = edges {
            let mut valid_edges = edges.iter().filter(|&e| e.is_valid()).collect::<Vec<_>>();
            if valid_edges.len() != grads.len() {
                panic!(
                    "Invalid number of gradients - expected {}, but got {}",
                    edges.len(),
                    grads.len()
                )
            }
            let mut new_grads = Vec::with_capacity(grads.len());
            let grad_len = grads.len();
            for i in (0..grad_len).rev() {
                let edge = valid_edges.pop().unwrap();
                if !edge.is_valid() {
                    continue;
                }
                let function = edge.function().unwrap().borrow();
                let metadata = function.input_metadata(edge.input_nr);
                // remove shrinks vector that's why can't use i so use 0 to always get first element.
                let grad = grads.pop().unwrap();
                if grad.sizes() != metadata.shape() {
                    if !util::is_expandable_to(metadata.shape(), grad.sizes()) {
                        panic!("invalid gradient at index {} - got {:?}, but expected shape comapatible with {:?}", i, grad.sizes(), metadata.shape());
                    }
                    new_grads.push(aten::sum_to(grad, metadata.shape()))
                } else {
                    new_grads.push(grad);
                }
            }
            new_grads.reverse();
            new_grads
        } else {
            grads
        }
    }

    pub fn call_function(func: *mut Node, inputs: InputBuffer) -> VariableList {
        let inputs = InputBuffer::variables(inputs);
        let fn_ = unsafe { &mut *func };
        let outputs = fn_.call(inputs);
        Self::validate_outputs(fn_.next_edges(), outputs)
    }

    pub fn evaluate_function(
        &mut self,
        graph_task: Rc<RefCell<GraphTask>>,
        func: Rc<RefCell<Node>>,
        inputs: InputBuffer,
    ) {
        let _fnc = func.as_ptr();
        let mut outputs = Self::call_function(func.as_ptr(), inputs);
        let fn_ = func.borrow_mut();
        let task = graph_task.borrow();
        let mut dependencies = task.dependencies.borrow_mut();
        // Get next_edges. if they exists then filter only valid edges. other wise return.
        let edges = fn_.next_edges().map(|e| {
            return e.iter().filter(|e| e.is_valid()).collect::<Vec<_>>();
        });
        if edges.is_none() {
            return;
        }
        let mut edges = edges.unwrap();
        assert!(edges.len() == outputs.len());
        while outputs.len() > 0 {
            let next = edges.pop().unwrap();
            if !next.is_valid() {
                continue;
            }
            let output = outputs.pop().unwrap();
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
                } else {
                    not_ready.insert(t, input_buffer);
                }
            }
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
            let outstanding_task = &graph_task.outstanding_tasks;
            outstanding_task.set(outstanding_task.get() - 1);
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
