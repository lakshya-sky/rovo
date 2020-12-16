use aten::native::{scalar_tensor, to};

use crate::{
    aten,
    c10::{
        get_default_dtype, kDouble, kFloat, kInt, kLong, type_meta_to_scalar_type, Device, Scalar,
        ScalarType, TensorOptions, KCPU,
    },
    core::NoGradGuard,
    tensor::Tensor,
};

#[derive(Debug, Copy, Clone, PartialEq)]
enum TensorDataContainerType {
    Scalar,
    InitList,
    Tensor,
}
#[inline(always)]
fn compute_desired_dtype(scalar_type: ScalarType) -> ScalarType {
    if scalar_type == kInt || scalar_type == kLong {
        // `tensor` with an integer type or a ` (nested) slice` / `vector`
        //  of integer types always produces a tensor of dtype `kLong`
        // (aka. i64).
        return kLong;
    } else if scalar_type == kFloat || scalar_type == kDouble {
        // `tensor` with a floating-point type or a `(nested) slice` / `vector` of
        //  floating-point types always produces a tensor of dtype `get_default_dtype()`
        return type_meta_to_scalar_type(&get_default_dtype());
    } else {
        return scalar_type;
    }
}
pub struct TensorDataContainer {
    type_: TensorDataContainerType,
    sizes: Vec<usize>,
    scalar_type: ScalarType,
    scalar: Scalar,
    tensor: Option<Tensor>,
}

impl TensorDataContainer {
    fn is_scalar(&self) -> bool {
        self.type_ == TensorDataContainerType::Scalar
    }

    fn is_tensor(&self) -> bool {
        self.type_ == TensorDataContainerType::Tensor
    }
    /*TensorDataContainer(uint8_t value)
        : sizes_(),
          scalar_type_(at::kByte),
          type_(TensorDataContainerType::Scalar),
          scalar_(value) {}
    */
    fn from_scalar(scalar: Scalar, scalar_type: ScalarType) -> Self {
        Self {
            sizes: vec![],
            scalar_type,
            type_: TensorDataContainerType::Scalar,
            scalar,
            tensor: None,
        }
    }

    /*
        TensorDataContainer(at::ArrayRef<float> values)
        : sizes_({(int64_t)values.size()}),
          scalar_type_(at::kFloat),
          type_(TensorDataContainerType::Tensor) {
      at::AutoNonVariableTypeMode non_var_type_mode(true);
      if (scalar_type_ == at::kBool) {
        tensor_ = at::tensor(values, at::TensorOptions().device(at::kCPU));
      } else {
        tensor_ = at::tensor(values, at::dtype(scalar_type_).device(at::kCPU));
      }
    }


    */
    fn from_slice<T>(values: &[T], scalar_type: ScalarType) -> Self {
        let tensor;
        if scalar_type == ScalarType::Bool {
            todo!()
        } else {
            tensor = aten::native::tensor(values, {
                let device: Device = KCPU.into();
                TensorOptions::with_dtype(scalar_type).set_device(device)
            })
        }
        Self {
            sizes: vec![values.len()],
            scalar_type,
            type_: TensorDataContainerType::Tensor,
            scalar: Scalar::default(),
            tensor: Some(tensor),
        }
    }

    fn fill_tensor(self, tensor: &Tensor) -> () {
        if self.is_scalar() {
            assert!(
                tensor.dim() == 0,
                "Expected a 0-dim Tensor, but got Tensor with dimensions: {}",
                tensor.dim()
            );
            let _guard = NoGradGuard::default();
            tensor.fill_(self.scalar);
        } else if self.is_tensor() {
            panic!(
                "TensorDataContainer is already a Tensor type, `fill_tensor` should not be called",
            );
        } else {
            panic!("Invalid TensorDataContainer type");
        }
    }

    pub fn convert_to_tensor(&self, mut options: TensorOptions) -> Tensor {
        if !options.has_dtype() {
            options.set_dtype_mut_(compute_desired_dtype(self.scalar_type));
        }
        if self.is_scalar() {
            return scalar_tensor(self.scalar, options);
        } else if self.is_tensor() {
            let output = to(self.tensor.as_ref().unwrap(), &options, false, false, None);
            return output;
        } else {
            panic!("Invalid TensorDataContainer type");
        }
    }
    /*
       else if (is_init_list()) {


      }
    }
      */
}

impl From<f32> for TensorDataContainer {
    fn from(val: f32) -> Self {
        TensorDataContainer::from_scalar(val.into(), ScalarType::Float)
    }
}

impl From<&[f32]> for TensorDataContainer {
    fn from(values: &[f32]) -> Self {
        Self::from_slice(values, ScalarType::Float)
    }
}
