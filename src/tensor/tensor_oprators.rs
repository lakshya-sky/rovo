use super::*;


trait Get<T> {
    type Output;
    fn get(&self, index: T) -> Self::Output;
}

impl Get<i64> for Tensor {
    type Output = Self;

    fn get(&self, index: i64) -> Self::Output {
        select(self, 0, index)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::autograd;
    #[test]
    fn test_get_indexing() {
        crate::init_rovo();
        let t = autograd::empty(&[3, 3], None, None);
        t.fill_(5.0);
        let result = t.get(0);
        dbg!(result);
    }
}
