#[macro_export]
macro_rules! AT_PRIVATE_CASE_TYPE{
    ($_ident: expr, $enum_type: path, $type: ty, $($args:expr),+)=>{
       if let $enum_type = $_ident {
            type Scalart = $type;
            return $($args),+()
        }
    }
}

#[macro_export]
macro_rules! AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2 {
    (_, _, $TYPE: expr, $name: expr, $($args:expr),+) => {{
        // match $TYPE {
        //     ScalarType::Int => {
        //         type Scalart = i32;
        //         $($args)*()
        //     },
        //     _ => todo!()
        // };
        $crate::AT_PRIVATE_CASE_TYPE!($TYPE, $crate::c10::ScalarType::Float, f32, $($args)+);
        $crate::AT_PRIVATE_CASE_TYPE!($TYPE, $crate::c10::ScalarType::Int, i32, $($args)+);
        $crate::AT_PRIVATE_CASE_TYPE!($TYPE, $crate::c10::ScalarType::Double, f64, $($args)+);
        $crate::AT_PRIVATE_CASE_TYPE!($TYPE, $crate::c10::ScalarType::Long, i64, $($args)+);
    }};
}

#[macro_export]
macro_rules! AT_DISPATCH_ALL_TYPES_AND {
    (_,$TYPE: expr, $name: expr, $($args:expr),+) => {{
        // match $TYPE {
        //     ScalarType::Int => {
        //         type Scalart = i32;
        //         $($args)*()
        //     },
        //     _ => todo!()
        // };
        $crate::AT_PRIVATE_CASE_TYPE!($TYPE, $crate::c10::ScalarType::Float, f32, $($args)+);
        $crate::AT_PRIVATE_CASE_TYPE!($TYPE, $crate::c10::ScalarType::Int, i32, $($args)+);
        $crate::AT_PRIVATE_CASE_TYPE!($TYPE, $crate::c10::ScalarType::Double, f64, $($args)+);
        $crate::AT_PRIVATE_CASE_TYPE!($TYPE, $crate::c10::ScalarType::Long, i64, $($args)+);
    }};
}

#[macro_export]
macro_rules! AT_DISPATCH_FLOATING_TYPES_AND2{
    (_,_,$TYPE: expr, $name: expr, $($args:expr),+)=>{{
        $crate::AT_PRIVATE_CASE_TYPE!($TYPE, $crate::c10::ScalarType::Float, f32, $($args)+);
        $crate::AT_PRIVATE_CASE_TYPE!($TYPE, $crate::c10::ScalarType::Double, f64, $($args)+);
    }}
}

#[macro_export]
macro_rules! AT_DISPATCH_FLOATING_TYPES{
    ($TYPE: expr, $name: expr, $($args:expr),+)=>{{
        $crate::AT_PRIVATE_CASE_TYPE!($TYPE, $crate::c10::ScalarType::Float, f32, $($args)+);
        $crate::AT_PRIVATE_CASE_TYPE!($TYPE, $crate::c10::ScalarType::Double, f64, $($args)+);
    }}
}

/*
[&] {
    const auto& the_type = scalar_type;
    at::ScalarType _st = ::detail::scalar_type(the_type);
    switch (_st) {
      case at::ScalarType::Double: {
        using scalar_t = double;
        return [&] {
          const auto min =
              static_cast<double>(std::numeric_limits<scalar_t>::lowest());
          const auto max =
              static_cast<double>(std::numeric_limits<scalar_t>::max());
          if ((__builtin_expect(
                  static_cast<bool>(!(from >= min && from <= max)), 0))) {
            throw ::c10::Error(
                {"_function_name_", "_file_name_", static_cast<uint32_t>(93)},
                ::c10::detail::if_empty_then(
                    ::c10::str("from", " is out of bounds for ", dtype),
                    "Expected "
                    "from >= min && from <= max"
                    " to be true, but got false.  "
                    "(Could this error message be improved?  If so, "
                    "please report an enhancement request to PyTorch.)"));
          };
          ;
          if ((__builtin_expect(
                  static_cast<bool>(!(to_inc >= min && to_inc <= max)), 0))) {
            throw ::c10::Error(
                {"_function_name_", "_file_name_", static_cast<uint32_t>(93)},
                ::c10::detail::if_empty_then(
                    ::c10::str("to - 1", " is out of bounds for ", dtype),
                    "Expected "
                    "to_inc >= min && to_inc <= max"
                    " to be true, but got false.  "
                    "(Could this error message be improved?  If so, "
                    "please report an enhancement request to PyTorch.)"));
          };
          ;
          constexpr auto digits = std::numeric_limits<scalar_t>::digits;
          if (from < -(1LL << digits) || from > (1LL << digits)) {
            ::c10::Warning::warn(
                {"_function_name_", "_file_name_", static_cast<uint32_t>(93)},
                ::c10::str(
                    "from",
                    " is out of bounds [-(2^",
                    digits,
                    "), 2^",
                    digits,
                    "]. ",
                    "Due to precision limitations ",
                    dtype,
                    " can support discrete uniform distribution only within this range. ",
                    "This warning will become an error in version 1.7 release, please fix the code in advance"),
                false);
          };
          if (to_inc < -(1LL << digits) || to_inc > (1LL << digits)) {
            ::c10::Warning::warn(
                {"_function_name_", "_file_name_", static_cast<uint32_t>(93)},
                ::c10::str(
                    "to - 1",
                    " is out of bounds [-(2^",
                    digits,
                    "), 2^",
                    digits,
                    "]. ",
                    "Due to precision limitations ",
                    dtype,
                    " can support discrete uniform distribution only within this range. ",
                    "This warning will become an error in version 1.7 release, please fix the code in advance"),
                false);
          };
        }();
      }
      case at::ScalarType::Float: {
        using scalar_t = float;
        return [&] {
          const auto min =
              static_cast<double>(std::numeric_limits<scalar_t>::lowest());
          const auto max =
              static_cast<double>(std::numeric_limits<scalar_t>::max());
          if ((__builtin_expect(
                  static_cast<bool>(!(from >= min && from <= max)), 0))) {
            throw ::c10::Error(
                {"_function_name_", "_file_name_", static_cast<uint32_t>(93)},
                ::c10::detail::if_empty_then(
                    ::c10::str("from", " is out of bounds for ", dtype),
                    "Expected "
                    "from >= min && from <= max"
                    " to be true, but got false.  "
                    "(Could this error message be improved?  If so, "
                    "please report an enhancement request to PyTorch.)"));
          };
          ;
          if ((__builtin_expect(
                  static_cast<bool>(!(to_inc >= min && to_inc <= max)), 0))) {
            throw ::c10::Error(
                {"_function_name_", "_file_name_", static_cast<uint32_t>(93)},
                ::c10::detail::if_empty_then(
                    ::c10::str("to - 1", " is out of bounds for ", dtype),
                    "Expected "
                    "to_inc >= min && to_inc <= max"
                    " to be true, but got false.  "
                    "(Could this error message be improved?  If so, "
                    "please report an enhancement request to PyTorch.)"));
          };
          ;
          constexpr auto digits = std::numeric_limits<scalar_t>::digits;
          if (from < -(1LL << digits) || from > (1LL << digits)) {
            ::c10::Warning::warn(
                {"_function_name_", "_file_name_", static_cast<uint32_t>(93)},
                ::c10::str(
                    "from",
                    " is out of bounds [-(2^",
                    digits,
                    "), 2^",
                    digits,
                    "]. ",
                    "Due to precision limitations ",
                    dtype,
                    " can support discrete uniform distribution only within this range. ",
                    "This warning will become an error in version 1.7 release, please fix the code in advance"),
                false);
          };
          if (to_inc < -(1LL << digits) || to_inc > (1LL << digits)) {
            ::c10::Warning::warn(
                {"_function_name_", "_file_name_", static_cast<uint32_t>(93)},
                ::c10::str(
                    "to - 1",
                    " is out of bounds [-(2^",
                    digits,
                    "), 2^",
                    digits,
                    "]. ",
                    "Due to precision limitations ",
                    dtype,
                    " can support discrete uniform distribution only within this range. ",
                    "This warning will become an error in version 1.7 release, please fix the code in advance"),
                false);
          };
        }();
      }
      case at::ScalarType::Half: {
        using scalar_t =
            decltype(c10::impl::ScalarTypeToCPPType<at::ScalarType::Half>::t);
        return [&] {
          const auto min =
              static_cast<double>(std::numeric_limits<scalar_t>::lowest());
          const auto max =
              static_cast<double>(std::numeric_limits<scalar_t>::max());
          if ((__builtin_expect(
                  static_cast<bool>(!(from >= min && from <= max)), 0))) {
            throw ::c10::Error(
                {"_function_name_", "_file_name_", static_cast<uint32_t>(93)},
                ::c10::detail::if_empty_then(
                    ::c10::str("from", " is out of bounds for ", dtype),
                    "Expected "
                    "from >= min && from <= max"
                    " to be true, but got false.  "
                    "(Could this error message be improved?  If so, "
                    "please report an enhancement request to PyTorch.)"));
          };
          ;
          if ((__builtin_expect(
                  static_cast<bool>(!(to_inc >= min && to_inc <= max)), 0))) {
            throw ::c10::Error(
                {"_function_name_", "_file_name_", static_cast<uint32_t>(93)},
                ::c10::detail::if_empty_then(
                    ::c10::str("to - 1", " is out of bounds for ", dtype),
                    "Expected "
                    "to_inc >= min && to_inc <= max"
                    " to be true, but got false.  "
                    "(Could this error message be improved?  If so, "
                    "please report an enhancement request to PyTorch.)"));
          };
          ;
          constexpr auto digits = std::numeric_limits<scalar_t>::digits;
          if (from < -(1LL << digits) || from > (1LL << digits)) {
            ::c10::Warning::warn(
                {"_function_name_", "_file_name_", static_cast<uint32_t>(93)},
                ::c10::str(
                    "from",
                    " is out of bounds [-(2^",
                    digits,
                    "), 2^",
                    digits,
                    "]. ",
                    "Due to precision limitations ",
                    dtype,
                    " can support discrete uniform distribution only within this range. ",
                    "This warning will become an error in version 1.7 release, please fix the code in advance"),
                false);
          };
          if (to_inc < -(1LL << digits) || to_inc > (1LL << digits)) {
            ::c10::Warning::warn(
                {"_function_name_", "_file_name_", static_cast<uint32_t>(93)},
                ::c10::str(
                    "to - 1",
                    " is out of bounds [-(2^",
                    digits,
                    "), 2^",
                    digits,
                    "]. ",
                    "Due to precision limitations ",
                    dtype,
                    " can support discrete uniform distribution only within this range. ",
                    "This warning will become an error in version 1.7 release, please fix the code in advance"),
                false);
          };
        }();
      }
      */
