use super::Tensor;
type DimVector = smallvec::SmallVec<[usize; 5]>;

// For now using Tensor type as OperandInfo, but when Tensor supports other types
// This needs to have a new struct similar to pytorch.

struct OperandInfo {
    is_output: bool,
    tensor: Tensor,
}

impl OperandInfo {
    pub fn new(tensor: Tensor) -> Self {
        Self {
            is_output: false,
            tensor,
        }
    }
}

// TensorIterator currently doesn't support the feature of having a Tensor
// both as an input and output.
#[derive(Default)]
pub struct TensorIterator {
    shape_: DimVector,
    oprands_: smallvec::SmallVec<[OperandInfo; 4]>,
    num_outputs_: usize,
    all_ops_same_shape_: bool,
    is_reduction_: bool,
}

impl TensorIterator {
    pub fn build(config: &TensorIteratorConfig) -> Self {
        let mut self_ = Self::default();
        self_.populate_operands(config);
        self_.compute_shape(config);
        self_.resize_outputs(config);
        self_
    }

    fn populate_operands(&mut self, config: &TensorIteratorConfig) {
        // Pytorch has both populate_operands and mark_tensor methods for filling
        // tensors in iterator, but here I am using only this method for both tasks.
        self.num_outputs_ = config.num_outputs_;
        for i in 0..config.tensors_.len() {
            self.oprands_
                .push(OperandInfo::new(config.tensors_[i].clone()));
        }

        // mark_outputs method is implemented here.
        for i in 0..self.num_outputs_ {
            self.oprands_[i].is_output = true;
        }
    }

    fn compute_shape(&mut self, config: &TensorIteratorConfig) {
        if let Some(static_shape_) = config.static_shape_.as_ref() {
            self.shape_ = static_shape_.clone();
        } else {
            self.all_ops_same_shape_ = true;
            let mut has_scalars = false;
            let mut has_tensors = false;
            for op in &(self.oprands_) {
                if config.resize_outputs_ && op.is_output {
                    continue;
                }
                let shape = op.tensor.sizes();
                if shape.len() == 0 {
                    has_scalars = true;
                } else {
                    has_tensors = true;
                }
                if has_scalars && has_tensors {
                    self.all_ops_same_shape_ = false;
                }
                if self.shape_.is_empty() {
                    self.shape_.extend_from_slice(shape);
                } else if shape != self.shape_.as_slice() {
                    self.all_ops_same_shape_ = false;
                    //Todo: infer_size
                    let tmp = super::tensor_util::infer_size(self.shape_.as_slice(), shape);
                    self.shape_.copy_from_slice(tmp.as_slice());
                }
                println!("Oprands have same shape? {}", self.all_ops_same_shape_);
            }
        }
    }

    pub fn resize_outputs(&self, config: &TensorIteratorConfig) {
        if config.static_shape_.is_none() {
            for i in 0..self.num_outputs_ {
                let tensor = &self.oprands_[i].tensor;
                assert_eq!(
                    tensor.sizes(),
                    self.shape_.as_slice(),
                    "Yet to implement output resizing for TenssorIterator"
                );
            }
        }
    }

    pub fn for_each(&self, op: impl Fn(&f64, &f64) -> f64) {
        // For each assumes that there is only single output right now
        // and the accepted closure accepts only two inputs

        let output = &self.oprands_.first().unwrap().tensor;
        let first_in = &self.oprands_.get(1).unwrap().tensor;
        let second_in = &self.oprands_.get(2).unwrap().tensor;
        let out_numel = output.numel();
        assert!(
            (out_numel == first_in.numel()) && (out_numel == second_in.numel()),
            "oprands should have same number of elements"
        );

        // let mut out_iter = output.get_tensor_impl().data.iter_mut();
        let mut first_iter = first_in.get_tensor_impl().data.iter();
        let mut second_iter = second_in.get_tensor_impl().data.iter();
        let sizes = output.sizes();

        // Todo: here it is assumed that each operand is only two dimentional
        // extend it to handle any dimensions.
        // let rows = sizes[0];
        let columns = sizes[1];
        let mut row;
        let mut col;
        let mut i = 0usize;
        loop {
            match (first_iter.next(), second_iter.next()) {
                (Some(x), Some(y)) => {
                    let result = op(x, y);
                    row = i / columns;
                    col = i % columns;
                    output.get_tensor_impl().data[[row, col]] = result;
                }
                _ => break,
            }
            i += 1;
        }
    }
}

pub struct TensorIteratorConfig {
    tensors_: smallvec::SmallVec<[Tensor; 4]>,
    num_inputs_: usize,
    num_outputs_: usize,
    static_shape_: Option<DimVector>,
    check_mem_overlap_: bool,
    allow_cpu_scalars_: bool,
    is_reduction_: bool,
    resize_outputs_: bool,
    check_all_same_dtype_: bool,
    check_all_same_device_: bool,
    enforce_safe_casting_to_output_: bool,
    promote_inputs_to_common_dtype_: bool,
    cast_common_dtypes_to_outputs_: bool,
}

impl Default for TensorIteratorConfig {
    fn default() -> Self {
        Self {
            tensors_: smallvec::smallvec![],
            num_inputs_: 0,
            num_outputs_: 0,
            static_shape_: None,
            check_mem_overlap_: false,
            allow_cpu_scalars_: false,
            is_reduction_: false,
            resize_outputs_: true,
            check_all_same_dtype_: true,
            check_all_same_device_: true,
            enforce_safe_casting_to_output_: false,
            promote_inputs_to_common_dtype_: false,
            cast_common_dtypes_to_outputs_: false,
        }
    }
}

impl TensorIteratorConfig {
    pub fn add_output(&mut self, output: &Tensor) -> &mut Self {
        assert!(self.num_inputs_ == 0);
        self.tensors_.push(output.clone());
        self.num_outputs_ += 1;
        return self;
    }

    pub fn add_input(&mut self, input: &Tensor) -> &mut Self {
        self.tensors_.push(input.clone());
        self.num_inputs_ += 1;
        return self;
    }

    pub fn build(&self) -> TensorIterator {
        TensorIterator::build(self)
    }
}
