/// PyTorch uses array of 256 bit internally for Vec256.
/// I am assuming that the array lives on stack rather than heap.
/// Since I can't any other way to create that array at compile time.
/// I am using const N for this.
/// The contraint for that is N = 32/size_of(T)
pub struct Vec256<T, const N: usize> {
    buf: [T; N],
}

impl<T, const N: usize> Vec256<T, N> {
    pub const fn size() -> usize {
        32 / std::mem::size_of::<T>()
    }
}
