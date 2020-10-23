#[derive(Clone)]
pub enum MemoryFormat {
    Contiguous,
    Preserve,
    ChannelsLast,
    ChannelsLast3d,
}
impl MemoryFormat {
    pub fn get_contiguous_memory_format() -> Self {
        Self::Contiguous
    }
}
