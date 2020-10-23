use super::{NewTensor, Tensor};
use crate::c10::MemoryFormat;
use std::{ffi::c_void, ptr::NonNull};
type DimVector = smallvec::SmallVec<[usize; 5]>;
type StrideVector = smallvec::SmallVec<[usize; 6]>;
type LOOP = fn(&[NonNull<u8>], &[usize], usize);
type LOOP2D = fn(&[NonNull<u8>], &[usize], usize, usize);
// For now using Tensor type as OperandInfo, but when Tensor supports other types
// This needs to have a new struct similar to pytorch.

struct DimCounter<'a, 'b> {
    shape: &'a [usize],
    range: &'b [usize],
    values: DimVector,
    offset: usize,
}

impl<'a, 'b> DimCounter<'a, 'b> {
    pub fn new(shape: &'a [usize], range: &'b [usize]) -> Self {
        let mut self_ = Self {
            shape,
            range,
            values: smallvec::smallvec![shape.len(); 0],
            offset: range[0],
        };
        let mut linear_offset = self_.range[0];
        let ndim = self_.values.len();
        for dim in 0..ndim {
            let size = self_.shape[dim];
            if size > 0 {
                self_.values[dim] = linear_offset % size;
                linear_offset /= size;
            }
        }
        assert_eq!(linear_offset, 0);
        self_
    }
    pub fn is_done(&self) -> bool {
        self.offset >= self.range[self.range.len() - 1]
    }
    pub fn increment(&mut self, step: &[usize; 2]) {
        self.offset += step[0] * step[1];
        let ndim = self.values.len();
        let mut overflow = step[0];
        let mut i = 0;
        if step[1] != 1 {
            assert!(step[0] == self.shape[0] && self.values[0] == 0);
            i = 1;
            overflow = step[1];
        }
        loop {
            if i >= ndim || overflow <= 0 {
                break;
            }
            let size = self.shape[i];
            let prev = self.values[i];
            let mut value = prev + overflow;
            if value >= size {
                overflow = 1;
                value -= size;
                assert!(value < size);
            } else {
                overflow = 0;
            }
            self.values[i] = value;
        }
        assert!(overflow == 0 || overflow == 1);
    }

    pub fn max_2d_step(&self) -> [usize; 2] {
        let step0 =
            (self.shape[0] - self.values[0]).min(self.range[self.range.len() - 1] - self.offset);
        let mut step1 = 1;
        if step0 == self.shape[0] && self.shape.len() >= 1 {
            step1 = (self.shape[1] - self.values[1])
                .min((self.range[self.range.len() - 1] - self.offset) / self.shape[0]);
        }
        [step0, step1]
    }
}
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
        self_.resize_inputs(config);
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
                    let tmp = super::tensor_util::infer_size(self.shape_.as_slice(), shape);
                    self.shape_.copy_from_slice(tmp.as_slice());
                }
                // eprintln!("Iter Shape: {:?}", self.shape_);
                // eprintln!("Oprands have same shape? {}", self.all_ops_same_shape_);
            }
        }
    }

    pub fn resize_outputs(&self, config: &TensorIteratorConfig) {
        if config.static_shape_.is_none() {
            for i in 0..self.num_outputs_ {
                let tensor = &self.oprands_[i].tensor;
                tensor.get_tensor_impl().data = tensor
                    .get_tensor_impl()
                    .data
                    .clone()
                    .into_shape(self.shape_.as_slice())
                    .unwrap();
                assert_eq!(
                    tensor.sizes(),
                    self.shape_.as_slice(),
                    "Yet to implement output resizing for TensorIterator"
                );
            }
        }
    }

    pub fn resize_inputs(&self, config: &TensorIteratorConfig) {
        if config.static_shape_.is_none() {
            for i in self.num_outputs_..self.oprands_.len() {
                let tensor = &self.oprands_[i].tensor;
                tensor.get_tensor_impl().data = tensor
                    .get_tensor_impl()
                    .data
                    .broadcast(self.shape_.as_slice())
                    .unwrap()
                    .into_owned();
                assert_eq!(
                    tensor.sizes(),
                    self.shape_.as_slice(),
                    "Yet to implement output resizing for TensorIterator"
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

    pub fn for_each_ternary(&self, op: impl Fn(&f64, &f64, &f64) -> f64) {
        // For each assumes that there is only single output right now
        // and the accepted closure accepts only two inputs

        let output = &self.oprands_.first().unwrap().tensor;
        let first_in = &self.oprands_.get(1).unwrap().tensor;
        let second_in = &self.oprands_.get(2).unwrap().tensor;
        let third_in = &self.oprands_.get(3).unwrap().tensor;
        let out_numel = output.numel();

        assert!(
            (out_numel == first_in.numel())
                && (out_numel == second_in.numel())
                && (out_numel == third_in.numel()),
            format!(
                "oprands should have same number of elements, but got {},{},{},{}",
                out_numel,
                first_in.numel(),
                second_in.numel(),
                third_in.numel()
            )
        );

        // let mut out_iter = output.get_tensor_impl().data.iter_mut();
        let mut first_iter = first_in.get_tensor_impl().data.iter();
        let mut second_iter = second_in.get_tensor_impl().data.iter();
        let mut third_iter = third_in.get_tensor_impl().data.iter();
        let sizes = output.sizes();

        // Todo: here it is assumed that each operand is only two dimentional
        // extend it to handle any dimensions.
        // let rows = sizes[0];
        let columns: usize = sizes[1];
        let mut row: usize;
        let mut col: usize;
        let mut idx = 0usize;
        {
            let data = &mut output.get_tensor_impl().data;
            loop {
                match (first_iter.next(), second_iter.next(), third_iter.next()) {
                    (Some(g), Some(i), Some(t)) => {
                        let result = op(g, i, t);
                        row = idx / columns;
                        col = idx % columns;
                        data[[row, col]] = result;
                    }
                    _ => break,
                }
                idx += 1;
            }
        }
    }

    pub fn nullary_op(output: &Tensor) -> Self {
        TensorIteratorConfig::default()
            .check_all_same_dtype(false)
            .add_output(output)
            .resize_outputs(false)
            .build()
    }

    pub fn output(&self) -> &Tensor {
        &self.oprands_.first().unwrap().tensor
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
        self
    }

    pub fn add_input(&mut self, input: &Tensor) -> &mut Self {
        self.tensors_.push(input.clone());
        self.num_inputs_ += 1;
        self
    }

    pub fn check_all_same_dtype(&mut self, check_all_same_dtype: bool) -> &mut Self {
        self.check_all_same_dtype_ = check_all_same_dtype;
        self
    }

    pub fn resize_outputs(&mut self, resize_outputs: bool) -> &mut Self {
        self.resize_outputs_ = resize_outputs;
        self
    }
    pub fn build(&self) -> TensorIterator {
        TensorIterator::build(self)
    }
}

// For now using Tensor type as OperandInfo, but when Tensor supports other types
// This needs to have a new struct similar to pytorch.

struct NewOperandInfo {
    is_output: bool,
    tensor: NewTensor,
    stride_bytes: StrideVector,
    data: Option<NonNull<c_void>>,
}

impl NewOperandInfo {
    pub fn new(tensor: NewTensor) -> Self {
        Self {
            is_output: false,
            tensor,
            stride_bytes: StrideVector::new(),
            data: None,
        }
    }
}
#[derive(PartialEq)]
enum FastSetupType {
    NONE,
    CONTIGUOUS,
    CHANNELSLAST,
    NONOVERLAPPINGDENSE,
}
// TensorIterator currently doesn't support the feature of having a Tensor
// both as an input and output.
#[derive(Default)]
pub struct NewTensorIterator {
    shape_: DimVector,
    operands_: smallvec::SmallVec<[NewOperandInfo; 4]>,
    num_outputs_: usize,
    all_ops_same_shape_: bool,
    is_reduction_: bool,
    has_coalesced_dimensions: bool,
}

impl NewTensorIterator {
    pub fn build(config: &NewTensorIteratorConfig) -> Self {
        let mut self_ = Self::default();
        self_.populate_operands(config);
        self_.compute_shape(config);
        self_.resize_outputs(config);
        if !self_.fast_setup(config) {
            todo!("Fast Setup failed. Implement normal setup")
        }
        for op in &mut self_.operands_ {
            op.data = Some(op.tensor.data_ptr());
        }
        self_
    }

    fn populate_operands(&mut self, config: &NewTensorIteratorConfig) {
        // Pytorch has both populate_operands and mark_tensor methods for filling
        // tensors in iterator, but here I am using only this method for both tasks.
        self.num_outputs_ = config.num_outputs_;
        for i in 0..config.tensors_.len() {
            self.operands_
                .push(NewOperandInfo::new(config.tensors_[i].clone()));
        }

        // mark_outputs method is implemented here.
        for i in 0..self.num_outputs_ {
            self.operands_[i].is_output = true;
        }
    }

    fn compute_shape(&mut self, config: &NewTensorIteratorConfig) {
        if let Some(static_shape_) = config.static_shape_.as_ref() {
            self.shape_ = static_shape_.clone();
        } else {
            self.all_ops_same_shape_ = true;
            let mut has_scalars = false;
            let mut has_tensors = false;
            for op in &(self.operands_) {
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
                    let tmp = super::tensor_util::infer_size(self.shape_.as_slice(), shape);
                    self.shape_.copy_from_slice(tmp.as_slice());
                }
                // eprintln!("Iter Shape: {:?}", self.shape_);
                // eprintln!("Oprands have same shape? {}", self.all_ops_same_shape_);
            }
        }
    }

    pub fn resize_outputs(&self, config: &NewTensorIteratorConfig) {
        if config.static_shape_.is_some() {
            return;
        }

        for i in 0..self.num_outputs_ {
            let tensor = &self.operands_[i].tensor;
            if tensor.sizes() != self.shape_.as_slice() {
                if config.resize_outputs_ {
                    let _ = tensor.resize(self.shape_.as_slice(), None);
                    continue;
                }

                assert!(
                    self.is_reduction_,
                    format!(
                        "output with shape {:?} doesn't match the broadcast shape {:?}",
                        tensor.sizes(),
                        self.shape_
                    )
                );
            }
        }
    }

    fn fast_setup(&mut self, config: &NewTensorIteratorConfig) -> bool {
        let setup_type = self.compute_fast_setup_type(config);
        if setup_type == FastSetupType::NONE {
            return false;
        }
        match setup_type {
            FastSetupType::CONTIGUOUS => {
                for i in 0..self.num_outputs_ {
                    let op = &self.operands_[i];
                    if !op.tensor.defined() {
                        panic!("Should create an empty tensor if not defined")
                    }
                }
            }
            _ => {}
        };
        if self.ndim() > 1 {
            self.has_coalesced_dimensions = true;
        }
        if self.ndim() >= 1 {
            self.shape_[0] = self.numel();
            self.shape_.resize(1, 0);
        }
        let ndim = self.ndim();
        for op in &mut self.operands_ {
            let element_size_in_bytes = op.tensor.element_size();
            op.stride_bytes.resize(ndim, 0);
            if ndim > 0 {
                op.stride_bytes[0] = element_size_in_bytes;
            }
        }
        return true;
    }

    fn ndim(&self) -> usize {
        self.shape_.len()
    }
    fn numel(&self) -> usize {
        self.shape_.iter().product()
    }

    fn compute_fast_setup_type(&self, _config: &NewTensorIteratorConfig) -> FastSetupType {
        if self.is_reduction_ || !self.all_ops_same_shape_ {
            return FastSetupType::NONE;
        }
        let mut is_contiguous = true;
        // let mut is_channels_last = true;
        // let mut is_non_overlapping_and_dense = true;
        for op in &self.operands_ {
            if op.tensor.defined() {
                is_contiguous &= op.tensor.is_contiguous(MemoryFormat::Contiguous);
                // is_contiguous &= op.tensor.is_contiguous(MemoryFormat::Contiguous);
                // is_contiguous &= op.tensor.is_contiguous(MemoryFormat::Contiguous);
            }
        }
        if is_contiguous {
            return FastSetupType::CONTIGUOUS;
        }

        todo!()
    }

    pub fn nullary_op(output: &NewTensor) -> Self {
        NewTensorIteratorConfig::default()
            .check_all_same_dtype(false)
            .add_output(output)
            .resize_outputs(false)
            .build()
    }

    pub fn output(&self) -> &NewTensor {
        &self.operands_.first().unwrap().tensor
    }

    pub fn ntensors(&self) -> usize {
        self.operands_.len()
    }
    fn get_strides(&self) -> StrideVector {
        let mut strides = StrideVector::new();
        for dim in 0..self.ndim() {
            for arg in 0..self.ntensors() {
                strides.push(self.operands_[arg].stride_bytes[dim]);
            }
        }
        strides
    }
    fn get_base_ptrs(&self) -> smallvec::SmallVec<[NonNull<u8>; 4]> {
        let mut ptrs = smallvec::SmallVec::<[NonNull<u8>; 4]>::new();
        for i in 0..self.ntensors() {
            ptrs.push(self.data_ptr(i).cast::<u8>());
        }
        ptrs
    }

    fn get_data_ptrs(
        &self,
        base: &smallvec::SmallVec<[NonNull<u8>; 4]>,
        counter: &[usize],
    ) -> smallvec::SmallVec<[NonNull<u8>; 4]> {
        let mut ptrs = smallvec::SmallVec::new();
        for dim in 0..self.ndim() {
            let value = counter[dim];
            for arg in 0..self.ntensors() {
                ptrs.insert(arg, unsafe {
                    NonNull::new_unchecked(
                        base[dim]
                            .as_ptr()
                            .add(value * self.operands_[arg].stride_bytes[dim]),
                    )
                })
            }
        }
        ptrs
    }
    fn data_ptr(&self, arg: usize) -> NonNull<c_void> {
        self.operands_[arg].data.unwrap().clone()
    }

    pub fn for_each<F>(&mut self, mut loop_: F)
    where
        F: FnMut(&[NonNull<u8>], &[usize], usize),
    {
        let ntensors = self.ntensors();

        let loop_2d = move |base: &[NonNull<u8>], strides: &[usize], size0: usize, size1: usize| {
            let mut data: smallvec::SmallVec<[NonNull<u8>; 4]> = base.into();
            let (strides, outer_strides) = strides.split_at(ntensors);
            for i in 0..size1 {
                if i > 0 {
                    for arg in 0..ntensors {
                        data[arg] = unsafe {
                            NonNull::new_unchecked(data[arg].as_ptr().add(outer_strides[arg]))
                        };
                    }
                }
                loop_(data.as_slice(), strides, size0);
            }
        };
        self.for_each_2d(loop_2d);
    }
    fn for_each_2d<F>(&mut self, loop_: F)
    where
        F: FnMut(&[NonNull<u8>], &[usize], usize, usize),
    {
        let numel = self.numel();
        if numel == 0 {
            return;
        } else if numel < 32768 {
            let t: Vec<usize> = (0..numel).collect();
            return self.serial_for_each(loop_, t.as_slice());
        }
    }
    fn serial_for_each<F>(&self, mut loop_: F, range: &[usize])
    where
        F: FnMut(&[NonNull<u8>], &[usize], usize, usize),
    {
        if range.len() == 0 {
            return;
        }
        let mut strides = self.get_strides();
        loop {
            if strides.len() >= 2 * self.ntensors() {
                break;
            }
            strides.push(0);
        }
        let base_ptrs = self.get_base_ptrs();
        if self.ndim() <= 1 {
            let ptrs = self.get_data_ptrs(&base_ptrs, &[range[0]]);
            loop_(ptrs.as_slice(), strides.as_slice(), range.len(), 1);
        } else {
            let mut counter = DimCounter::new(self.shape_.as_slice(), range);
            loop {
                if counter.is_done() {
                    break;
                }
                let ptrs = self.get_data_ptrs(&base_ptrs, counter.values.as_slice());
                let step = counter.max_2d_step();
                loop_(ptrs.as_slice(), strides.as_slice(), step[0], step[1]);
                counter.increment(&step);
            }
        }
    }
}

pub struct NewTensorIteratorConfig {
    tensors_: smallvec::SmallVec<[NewTensor; 4]>,
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

impl Default for NewTensorIteratorConfig {
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

impl NewTensorIteratorConfig {
    pub fn add_output(&mut self, output: &NewTensor) -> &mut Self {
        assert!(self.num_inputs_ == 0);
        self.tensors_.push(output.clone());
        self.num_outputs_ += 1;
        self
    }

    pub fn add_input(&mut self, input: &NewTensor) -> &mut Self {
        self.tensors_.push(input.clone());
        self.num_inputs_ += 1;
        self
    }

    pub fn check_all_same_dtype(&mut self, check_all_same_dtype: bool) -> &mut Self {
        self.check_all_same_dtype_ = check_all_same_dtype;
        self
    }

    pub fn resize_outputs(&mut self, resize_outputs: bool) -> &mut Self {
        self.resize_outputs_ = resize_outputs;
        self
    }
    pub fn build(&self) -> NewTensorIterator {
        NewTensorIterator::build(self)
    }
}
