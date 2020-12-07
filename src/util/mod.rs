mod bitset;
mod ordered_dict;
mod vec256;

pub use bitset::*;
pub use ordered_dict::*;
pub use vec256::*;

pub fn is_expandable_to(shape: &[usize], desired: &[usize]) -> bool {
    let shape_dim = shape.len();
    let desired_dim = desired.len();
    if shape_dim > desired_dim {
        return false;
    }
    let mut i = 0usize;
    loop {
        if i >= shape_dim {
            break;
        }
        let size = shape[shape_dim - i - 1];
        let target = desired[desired_dim - i - 1];
        if size != target && size != 1 {
            return false;
        }
        i += 1;
    }
    return true;
}
