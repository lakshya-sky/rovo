#[derive(Debug, Clone)]
pub enum Layout {
    Strided,
    Sparse,
}

pub const K_STRIDED: Layout = Layout::Strided;
pub const K_SPARSE: Layout = Layout::Sparse;
