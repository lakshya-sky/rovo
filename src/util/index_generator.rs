pub struct IndexGenerator {
    start: usize,
    len: usize,
}

impl IndexGenerator {
    pub fn new() -> Self {
        Self { start: 0, len: 0 }
    }

    pub fn next(&mut self) -> usize {
        let start = self.start;
        self.start += 1;
        self.len += 1;
        start
    }

    pub fn len(&self) -> usize {
        self.len
    }
}
