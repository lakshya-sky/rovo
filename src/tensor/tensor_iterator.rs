use super::Tensor;
use crate::aten;
use crate::autograd;
use crate::c10::{can_cast, elementSize, Device, DeviceType, ScalarType, TensorOptions, KCPU};

use std::{ffi::c_void, ops::Range, ptr::NonNull};
type DimVector = smallvec::SmallVec<[usize; 5]>;
type StrideVector = smallvec::SmallVec<[usize; 6]>;
type LOOP = fn(&[NonNull<u8>], &[usize], usize);
type LOOP2D = fn(&[NonNull<u8>], &[usize], usize, usize);
// For now using Tensor type as OperandInfo, but when Tensor supports other types
// This needs to have a new struct similar to pytorch.

struct DimCounter<'a> {
    shape: &'a [usize],
    range: Range<usize>,
    values: DimVector,
    offset: usize,
}

impl<'a> DimCounter<'a> {
    pub fn new(shape: &'a [usize], range: Range<usize>) -> Self {
        let mut self_ = Self {
            shape,
            values: smallvec::smallvec![0; shape.len()],
            offset: range.start,
            range,
        };
        let mut linear_offset = self_.range.start;
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
        self.offset >= self.range.end
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
        let step0 = (self.shape[0] - self.values[0]).min(self.range.end - self.offset);
        let mut step1 = 1;
        if step0 == self.shape[0] && self.shape.len() >= 1 {
            step1 = (self.shape[1] - self.values[1])
                .min((self.range.end + 1 - self.offset) / self.shape[0]);
            // range[..]+1 is used to get one element greater than last element because c++ uses range instead of slice where range.end will give this behavior.
        }
        [step0, step1]
    }
}

// For now using Tensor type as OperandInfo, but when Tensor supports other types
// This needs to have a new struct similar to pytorch.

#[derive(Debug)]
struct NewOperandInfo {
    is_output: bool,
    is_read_write: bool,
    tensor: Tensor,
    stride_bytes: StrideVector,
    data: Option<NonNull<c_void>>,
    device: Device,
    target_dtype: ScalarType,
    current_dtype: ScalarType,
}

impl NewOperandInfo {
    pub fn new(tensor: Tensor) -> Self {
        if tensor.defined() {
            let target_dtype = tensor.scalar_type();
            Self {
                is_output: false,
                is_read_write: false,
                tensor,
                stride_bytes: StrideVector::new(),
                data: None,
                target_dtype: target_dtype,
                current_dtype: target_dtype,
                device: Device::default(),
            }
        } else {
            Self {
                is_output: false,
                is_read_write: false,
                tensor,
                stride_bytes: StrideVector::new(),
                data: None,
                target_dtype: ScalarType::Undefined,
                current_dtype: ScalarType::Undefined,
                device: Device::default(),
            }
        }
    }
    pub fn is_type_defined(&self) -> bool {
        self.target_dtype != ScalarType::Undefined
    }
    pub fn options(&self) -> TensorOptions {
        let mut op = TensorOptions::default().set_dtype_(self.target_dtype);
        op.set_device_mut(self.device.clone());
        op
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
pub struct TensorIterator {
    shape_: DimVector,
    operands_: smallvec::SmallVec<[NewOperandInfo; 4]>,
    num_outputs_: usize,
    all_ops_same_shape_: bool,
    is_reduction_: bool,
    has_coalesced_dimensions: bool,
    common_dtype_: ScalarType,
    perm: DimVector,
}

impl TensorIterator {
    fn reorder_dimensions(&mut self, _config: &TensorIteratorConfig) {
        let ndim = self.ndim();
        self.perm.resize(ndim, 0);
        if ndim == 1 {
            self.perm[0] = 0;
            return;
        }
        // initialize perm with n-1, n-2, ..., 1, 0
        // let n = self.perm.len();
        let mut val = 0;
        for i in self.perm.iter_mut().rev() {
            *i = val;
            val += 1;
        }
        // returns 1 if the dim0 should come after dim1, -1 if dim0 should come
        // before dim1, and 0 if the comparison is ambiguous.
        let ntensors = self.ntensors();
        let stride_bytes: Vec<_> = self
            .operands_
            .iter()
            .map(|i| i.stride_bytes.clone())
            .collect();
        let is_reduction = self.is_reduction_;
        let is_output: Vec<_> = self.operands_.iter().map(|i| i.is_output).collect();
        let should_swap = |dim0: usize, dim1: usize| {
            let ret = 0;
            for arg in 0..ntensors {
                if stride_bytes[arg].is_empty() {
                    continue;
                }
                let stride0 = stride_bytes[arg][dim0];
                let stride1 = stride_bytes[arg][dim1];
                if is_reduction && is_output[arg] {
                    // move reduced dimensions to the front
                    if (stride0 == 0) != (stride1 == 0) {
                        return if stride1 == 0 { 1 } else { -1 };
                    }
                }
                if stride0 == 0 || stride1 == 0 {
                    continue;
                } else if stride0 <= stride1 {
                    return -1;
                } else {
                    return 1;
                }
            }
            return ret;
        };

        // insertion sort with support for ambiguous comparisons
        for i in 1..self.ndim() {
            let mut dim1 = i;
            for dim0 in (0..i - 1).rev() {
                let comparison = should_swap(self.perm[dim0], self.perm[dim1]);
                if comparison > 0 {
                    self.perm.swap(dim0, dim1);
                    dim1 = dim0;
                } else if comparison < 0 {
                    break;
                }
            }
        }
        // perform re-ordering of shape and strides
        self.permute_dimensions();
    }

    pub fn build(config: &TensorIteratorConfig) -> Self {
        let mut self_ = Self::default();
        self_.populate_operands(config);
        self_.mark_outputs();
        self_.compute_mem_overlaps(config);

        self_.compute_shape(config);
        self_.resize_outputs(config);
        self_.compute_types(config);
        if !self_.fast_setup(config) {
            self_.compute_strides(config);
            self_.reorder_dimensions(config);
            self_.allocate_outputs();
            self_.coalesce_dimensions();
        }
        for op in &mut self_.operands_ {
            op.data = Some(op.tensor.data_ptr());
        }
        // dbg!(&self_.operands_.len());
        self_
    }
    fn compute_types(&mut self, config: &TensorIteratorConfig) {
        let common_device = KCPU;
        self.common_dtype_ = ScalarType::Undefined;
        let mut output_dtype = ScalarType::Undefined;
        let mut has_different_input_dtypes = false;
        let mut has_different_output_dtypes = false;
        let mut has_undefined_outputs = false;
        for op in self.operands_.iter_mut() {
            if !op.is_type_defined() {
                assert!(op.is_output, "Found type undefined input tensor!");
                if let Some(dtype_and_device) = config.static_dtype_and_device_.as_ref() {
                    op.target_dtype = dtype_and_device.0;
                    op.device = dtype_and_device.1.clone();
                } else {
                    assert!(config.check_all_same_device_);
                    has_undefined_outputs = true;
                    continue;
                }
            }
            if !op.tensor.defined() {
                assert!(op.is_output, "Found undefined input tensor!");
                continue;
            }
            assert_eq!(op.target_dtype, op.current_dtype);
            if !op.is_output && op.target_dtype != self.common_dtype_ {
                if self.common_dtype_ == ScalarType::Undefined {
                    self.common_dtype_ = op.target_dtype;
                } else {
                    has_different_input_dtypes = true;
                }
            } else if op.is_output && op.target_dtype != self.common_dtype_ {
                if output_dtype == ScalarType::Undefined {
                    output_dtype = op.target_dtype;
                } else {
                    has_different_output_dtypes = true;
                }
            }
        }
        assert!(
            !(has_different_input_dtypes
                && !config.promote_inputs_to_common_dtype_
                && (has_undefined_outputs
                    || config.enforce_safe_casting_to_output_
                    || config.cast_common_dtype_to_outputs_)),
        );
        if config.check_all_same_dtype_
            && (has_different_input_dtypes
                || has_different_output_dtypes
                || (self.common_dtype_ != output_dtype && output_dtype != ScalarType::Undefined))
        {
            // Throws an informative error message
            for op in self.operands_.iter() {
                if !op.tensor.defined() {
                    continue;
                }

                assert!(
                    op.target_dtype == self.common_dtype_,
                    format!(
                        "Found dtype {:?} but expected {:?}",
                        op.target_dtype, self.common_dtype_
                    )
                );
            }
        }
        if !has_undefined_outputs
            && !config.check_all_same_device_
            && !config.promote_inputs_to_common_dtype_
            && !config.cast_common_dtype_to_outputs_
            && !config.enforce_safe_casting_to_output_
        {
            // Invalidates common_dtype_ if it could not be inferred
            self.common_dtype_ = if has_different_input_dtypes {
                ScalarType::Undefined
            } else {
                self.common_dtype_
            };
            return;
        }
        if has_different_input_dtypes && config.promote_inputs_to_common_dtype_ {
            self.common_dtype_ = self.compute_common_dtype();
        }
        // let mut max_cpu_scalars_on_cuda = if config.allow_cpu_scalars_ { 1 } else { 0 };
        // let mut current_cpu_scalars_on_cuda = 0;
        for op in self.operands_.iter_mut() {
            if !op.is_type_defined() {
                op.target_dtype = self.common_dtype_;
                op.device = common_device.into();
                continue;
            }

            // Skips undefined tensors
            if !op.tensor.defined() {
                continue;
            }

            // Checks safe casting, if requested
            if config.enforce_safe_casting_to_output_
                && op.is_output
                && op.current_dtype != self.common_dtype_
            {
                assert!(
                    can_cast(self.common_dtype_, op.current_dtype),
                    format!(
                        "result type {:?} can't be cast to the desired output type {:?}",
                        self.common_dtype_, op.current_dtype
                    )
                );
            }
        }
    }
    fn compatible_stride(&self, element_size: usize) -> StrideVector {
        let mut stride = StrideVector::new();
        let mut next_stride = element_size;
        for dim in 0..self.ndim() {
            stride.push(next_stride);
            next_stride *= self.shape_[dim];
        }
        return stride;
    }
    fn invert_perm(&self, input: &[usize]) -> DimVector {
        // Invert the permutation caused by reorder_dimensions. This is not valid
        // after coalesce_dimensions is called.
        assert!(!self.has_coalesced_dimensions);
        assert!(input.len() == self.perm.len());
        let mut res = smallvec::smallvec![0; input.len()]; //no initialization needed, every value in res should be written to.
        for dim in 0..self.ndim() {
            res[self.perm[dim]] = input[dim];
        }
        return res;
    }

    fn allocate_outputs(&mut self) {
        let ndim = self.ndim();
        let mut stride_bytes = vec![];
        let mut tensor_shape = vec![];
        let mut inverted_vec = vec![true; self.num_outputs_];
        for i in 0..self.num_outputs_ {
            {
                let op = &self.operands_[i];
                if !op.tensor.defined() {
                    assert!(op.is_type_defined(), "No type for operand {}", i);
                    let el_size = elementSize(op.target_dtype);
                    //Handle for last_* conditions;
                    stride_bytes.push(self.compatible_stride(el_size));
                }
            }
            {
                let op = &mut self.operands_[i];
                if !op.tensor.defined() {
                    op.stride_bytes = stride_bytes[i].clone();
                }
            }

            {
                let op = &self.operands_[i];
                if !op.tensor.defined() {
                    let mut inverted = true;
                    for dim in 1..=ndim {
                        if self.perm[dim - 1] != (ndim - dim) {
                            inverted = false;
                            break;
                        }
                    }
                    inverted_vec[i] = inverted;
                    tensor_shape.push(self.invert_perm(&self.shape_.as_slice()));
                }
            }
            {
                let op = &mut self.operands_[i];
                if !op.tensor.defined() {
                    if inverted_vec[i] {
                        op.tensor.move_tensor(aten::native::empty(
                            tensor_shape[i].as_slice(),
                            op.options(),
                            None,
                        ));
                    } else {
                        // let tensor_stride = self.invert_perm(&op.stride_bytes.as_slice());
                        // for dim in 0..ndim {
                        //     tensor_stride[dim] /= el_size;
                        // }
                        // op.tensor =
                        //     aten::native::empty_strided(tensor_shape, tensor_stride, op.options());
                        todo!()
                    }
                    op.current_dtype = op.target_dtype;
                }
            }
        }
    }

    fn coalesce_dimensions(&mut self) {
        let ndim = self.ndim();
        if ndim <= 1 {
            return;
        }
        let shape_ = self.shape_.clone();
        let ntensors = self.ntensors();
        let stride_bytes: Vec<_> = self
            .operands_
            .iter()
            .map(|i| i.stride_bytes.clone())
            .collect();
        let can_coalesce = |dim0: usize, dim1: usize| {
            let shape0 = shape_[dim0];
            let shape1 = shape_[dim1];
            if shape0 == 1 || shape1 == 1 {
                return true;
            }
            for i in 0..ntensors {
                let stride = &stride_bytes[i];
                if shape0 * stride[dim0] != stride[dim1] {
                    return false;
                }
            }
            return true;
        };
        // replace each operands stride at dim0 with its stride at dim1
        let replace_stride = |itr: &mut Self, dim0: usize, dim1: usize| {
            for i in 0..ntensors {
                let stride = &mut itr.operands_[i].stride_bytes;
                stride[dim0] = stride[dim1];
            }
        };

        let mut prev_dim = 0;
        for dim in 1..ndim {
            if can_coalesce(prev_dim, dim) {
                if self.shape_[prev_dim] == 1 {
                    replace_stride(self, prev_dim, dim);
                }
                self.shape_[prev_dim] *= self.shape_[dim];
            } else {
                prev_dim += 1;
                if prev_dim != dim {
                    replace_stride(self, prev_dim, dim);
                    self.shape_[prev_dim] = self.shape_[dim];
                }
            }
        }

        self.shape_.resize(prev_dim + 1, 0);
        for i in 0..self.ntensors() {
            self.operands_[i].stride_bytes.resize(ndim, 0);
        }
        self.has_coalesced_dimensions = true;
    }

    fn compute_common_dtype(&mut self) -> ScalarType {
        let mut state = aten::native::ResultTypeState::default();
        for op in self.operands_.iter() {
            if op.is_output {
                continue;
            }
            state = aten::native::update_result_type_state(&op.tensor, state);
        }
        self.common_dtype_ = aten::native::result_type(&state);
        assert!(self.common_dtype_ != ScalarType::Undefined);
        self.common_dtype_
    }
    fn populate_operands(&mut self, config: &TensorIteratorConfig) {
        // Pytorch has both populate_operands and mark_tensor methods for filling
        // tensors in iterator, but here I am using only this method for both tasks.

        for tensor in config.tensors_.iter() {
            self.operands_.push(NewOperandInfo::new(tensor.clone()))
        }
        self.num_outputs_ = config.num_outputs_;
    }

    fn mark_outputs(&mut self) {
        // Here I used three loops because, rust doesn't let us use mutable and immutable
        // references at the same time.
        for i in 0..self.num_outputs_ {
            self.operands_[i].is_output = true;
        }

        let ntensors = self.ntensors();
        let mut indices = vec![];
        for i in 0..self.num_outputs_ {
            let output = &(&self.operands_[i].tensor);
            if !output.defined() {
                continue;
            }
            for arg in self.num_outputs_..ntensors {
                let input = &self.operands_[arg].tensor;
                if output.is_same(input) {
                    indices.push(i);
                }
            }
        }
        for i in indices {
            self.operands_[i].is_read_write = true;
        }
    }
    fn compute_mem_overlaps(&mut self, config: &TensorIteratorConfig) {
        if !config.check_mem_overlap_ {
            return;
        }
        for i in 0..self.num_outputs_ {
            let output = &self.operands_[i].tensor;
            if !output.defined() {
                continue;
            }
            aten::assert_no_internal_overlap(output);
            for j in self.num_outputs_..self.ntensors() {
                let input = &self.operands_[j].tensor;
                aten::assert_no_partial_overlap(output, input);
            }
        }
    }

    fn compute_shape(&mut self, config: &TensorIteratorConfig) {
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

    pub fn resize_outputs(&self, config: &TensorIteratorConfig) {
        if config.static_shape_.is_some() {
            return;
        }

        for i in 0..self.num_outputs_ {
            let tensor = &self.operands_[i].tensor;
            if tensor.defined() && tensor.sizes() != self.shape_.as_slice() {
                if config.resize_outputs_ {
                    tensor.resize(self.shape_.as_slice(), None);
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

    fn compute_strides(&mut self, config: &TensorIteratorConfig) {
        let ndim = self.ndim();
        for op in self.operands_.iter_mut() {
            if op.tensor.defined() {
                let origional_shape = if config.static_shape_.is_some() {
                    self.shape_.as_slice()
                } else {
                    op.tensor.sizes()
                };
                let original_stride = op.tensor.strides();
                let element_size_in_bytes = op.tensor.element_size();
                let offset = ndim - origional_shape.len();
                if offset > 0 {
                    op.stride_bytes.resize(ndim, 0);
                } else {
                    op.stride_bytes.resize(ndim, 0);
                }
                for i in 0..origional_shape.len() {
                    if origional_shape[i] == 1 {
                        op.stride_bytes[offset + i] = 0;
                    } else {
                        op.stride_bytes[offset + i] = original_stride[i] * element_size_in_bytes;
                    }
                }
            }
        }
    }
    fn fast_setup(&mut self, config: &TensorIteratorConfig) -> bool {
        let setup_type = self.compute_fast_setup_type(config);
        if setup_type == FastSetupType::NONE {
            return false;
        }
        match setup_type {
            FastSetupType::CONTIGUOUS => {
                for i in 0..self.num_outputs_ {
                    let op = &mut self.operands_[i];
                    if !op.tensor.defined() {
                        assert!(op.is_type_defined(), "No type for operand {}", i);
                        op.tensor
                            .move_tensor(autograd::empty(self.shape_.as_slice(), None, None));
                        op.current_dtype = op.target_dtype;
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
    pub fn numel(&self) -> usize {
        self.shape_.iter().product()
    }

    fn compute_fast_setup_type(&self, _config: &TensorIteratorConfig) -> FastSetupType {
        if self.is_reduction_ || !self.all_ops_same_shape_ {
            return FastSetupType::NONE;
        }
        let mut is_contiguous = true;
        // let mut is_channels_last = true;
        // let mut is_non_overlapping_and_dense = true;
        for op in &self.operands_ {
            if op.tensor.defined() {
                is_contiguous &= op.tensor.is_contiguous();
                // is_contiguous &= op.tensor.is_contiguous(MemoryFormat::Contiguous);
                // is_contiguous &= op.tensor.is_contiguous(MemoryFormat::Contiguous);
            }
        }
        if is_contiguous {
            return FastSetupType::CONTIGUOUS;
        }
        FastSetupType::NONE
    }

    pub fn nullary_op(output: &Tensor) -> Self {
        TensorIteratorConfig::default()
            .check_all_same_dtype(false)
            .add_output(output)
            .resize_outputs(false)
            .build()
    }

    pub fn unary_op(out: &Tensor, a: &Tensor, check_mem_overlap: bool) -> Self {
        TensorIteratorConfig::default()
            .set_check_mem_overlap(check_mem_overlap)
            .add_output(out)
            .add_input(a)
            .cast_common_dtype_to_outputs(false)
            .enforce_safe_casting_to_output(false)
            .check_all_same_dtype(true)
            .build()
    }

    pub fn binary_op(out: &Tensor, a: &Tensor, b: &Tensor, check_mem_overlap: bool) -> Self {
        TensorIteratorConfig::default()
            .set_check_mem_overlap(check_mem_overlap)
            .add_output(out)
            .add_input(a)
            .add_input(b)
            .allow_cpu_scalars(true)
            .promote_inputs_to_common_dtype(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(true)
            .build()
    }
    pub fn output(&self) -> &Tensor {
        &self.operands_.first().unwrap().tensor
    }

    pub fn ntensors(&self) -> usize {
        self.operands_.len()
    }
    pub fn dtype(&self) -> ScalarType {
        self.dtype_(0)
    }
    pub fn dtype_(&self, i: usize) -> ScalarType {
        self.operands_[i].current_dtype
    }
    pub fn device(&self, arg: usize) -> &Device {
        &self.operands_[arg].device
    }

    pub fn device_type(&self, arg: usize) -> DeviceType {
        self.device(arg).type_()
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
                        base[arg]
                            .as_ptr()
                            .add(value * self.operands_[arg].stride_bytes[dim]),
                    )
                })
            }
        }
        ptrs
    }
    fn permute_dimensions(&mut self) {
        let perm = self.perm.as_slice();
        assert!(perm.len() == self.ndim());

        let reorder = |data: &[usize]| {
            let mut res: DimVector = (0..data.len()).map(|_| 0).collect();
            for i in 0..perm.len() {
                res[i] = data[perm[i]];
            }
            return res;
        };

        // Update shape and strides
        self.shape_ = reorder(self.shape_.as_slice());
        for op in self.operands_.iter_mut() {
            if op.stride_bytes.len() > 0 {
                op.stride_bytes = reorder(op.stride_bytes.as_slice()).as_slice().into();
            }
        }
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
            let range = 0..numel;
            return self.serial_for_each_2d(loop_, range);
        }
    }

    pub fn serial_for_each<F>(&self, mut loop_: F, range: Range<usize>)
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
        self.serial_for_each_2d(loop_2d, range);
    }

    fn serial_for_each_2d<F>(&self, mut loop_: F, range: Range<usize>)
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
            let ptrs = self.get_data_ptrs(&base_ptrs, &[range.start]);
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

pub struct TensorIteratorConfig {
    tensors_: smallvec::SmallVec<[Tensor; 4]>,
    num_inputs_: usize,
    num_outputs_: usize,
    static_shape_: Option<DimVector>,
    static_dtype_and_device_: Option<(ScalarType, Device)>,
    check_mem_overlap_: bool,
    allow_cpu_scalars_: bool,
    is_reduction_: bool,
    resize_outputs_: bool,
    check_all_same_dtype_: bool,
    check_all_same_device_: bool,
    enforce_safe_casting_to_output_: bool,
    promote_inputs_to_common_dtype_: bool,
    cast_common_dtype_to_outputs_: bool,
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
            cast_common_dtype_to_outputs_: false,
            static_dtype_and_device_: None,
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
    pub fn set_check_mem_overlap(&mut self, check_mem_overlap: bool) -> &mut Self {
        self.check_mem_overlap_ = check_mem_overlap;
        self
    }
    pub fn check_all_same_dtype(&mut self, check_all_same_dtype: bool) -> &mut Self {
        self.check_all_same_dtype_ = check_all_same_dtype;
        self
    }
    pub fn promote_inputs_to_common_dtype(
        &mut self,
        promote_inputs_to_common_dtype: bool,
    ) -> &mut Self {
        self.promote_inputs_to_common_dtype_ = promote_inputs_to_common_dtype;
        self
    }

    pub fn allow_cpu_scalars(&mut self, allow_cpu_scalars: bool) -> &mut Self {
        self.allow_cpu_scalars_ = allow_cpu_scalars;
        self
    }
    pub fn cast_common_dtype_to_outputs(
        &mut self,
        cast_common_dtype_to_outputs: bool,
    ) -> &mut Self {
        self.cast_common_dtype_to_outputs_ = cast_common_dtype_to_outputs;
        if cast_common_dtype_to_outputs {
            self.check_all_same_dtype_ = false;
        }
        self
    }
    pub fn enforce_safe_casting_to_output(
        &mut self,
        enforce_safe_casting_to_output: bool,
    ) -> &mut Self {
        self.enforce_safe_casting_to_output_ = enforce_safe_casting_to_output;
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
