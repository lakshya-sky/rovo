use std::ops::Shl;

type _WordT = u64;
const BITS_PER_WORD: usize = u64::BITS as usize;

struct BaseBitSet<const N: usize> {
    w: [_WordT; N],
}

impl<const N: usize> Default for BaseBitSet<N> {
    fn default() -> Self {
        let w = [0; N];
        Self { w }
    }
}

impl<const N: usize> BaseBitSet<N> {
    pub fn new(val: _WordT) -> Self {
        let mut w = [0; N];
        w[0] = val;
        BaseBitSet { w }
    }
    pub fn whichword(pos: usize) -> usize {
        pos / BITS_PER_WORD
    }
    pub fn whichbit(pos: usize) -> usize {
        pos % BITS_PER_WORD
    }

    pub fn maskbit(pos: usize) -> _WordT {
        (1 as _WordT).shl(Self::whichbit(pos))
    }
    pub fn getword(&self, pos: usize) -> _WordT {
        self.w[Self::whichword(pos) as usize]
    }
    pub fn getword_mut(&mut self, pos: usize) -> &mut _WordT {
        self.w.get_mut(Self::whichword(pos) as usize).unwrap()
    }
    pub fn do_flip(&mut self) {
        for i in 0..N {
            self.w[i] = !self.w[i]
        }
    }
}

type Base<const N: usize> = BaseBitSet<N>;

struct CommonFns;
impl CommonFns {
    fn whichbit(pos: usize) -> usize {
        pos % BITS_PER_WORD
    }

    fn maskbit(pos: usize) -> _WordT {
        (1 as _WordT).shl(Self::whichbit(pos))
    }
}
struct Reference {
    wp: *mut _WordT,
    bpos: usize,
}

impl Reference {
    pub fn new<const N: usize>(b: &mut BitSet<N>, pos: usize) -> Self {
        Self {
            wp: b.getword_mut(pos),
            bpos: CommonFns::whichbit(pos),
        }
    }
    pub unsafe fn assign_value(&mut self, x: bool) {
        if x {
            *self.wp |= CommonFns::maskbit(self.bpos);
        } else {
            *self.wp &= !CommonFns::maskbit(self.bpos);
        }
    }

    pub unsafe fn flip(&mut self) -> &mut Self {
        *self.wp ^= CommonFns::maskbit(self.bpos);
        self
    }
}

impl std::ops::Not for Reference {
    type Output = bool;
    fn not(self) -> Self::Output {
        (unsafe { *self.wp } & CommonFns::maskbit(self.bpos)) == 0
    }
}
#[derive(Default)]
pub struct BitSet<const N: usize> {
    base: BaseBitSet<N>,
}

impl<const N: usize> BitSet<N> {
    pub fn new(val: u64) -> Self {
        Self {
            base: Base::new(val),
        }
    }

    pub fn getword(&self, pos: usize) -> _WordT {
        self.base.getword(pos)
    }
    pub fn getword_mut(&mut self, pos: usize) -> &mut _WordT {
        self.base.getword_mut(pos)
    }

    pub fn unchecked_set(&mut self, position: usize, val: bool) {
        if val {
            *self.getword_mut(position) |= CommonFns::maskbit(position);
        } else {
            *self.getword_mut(position) &= !CommonFns::maskbit(position);
        }
    }
    pub fn unchecked_flip(&mut self, position: usize) {
        *self.getword_mut(position) ^= CommonFns::maskbit(position);
    }

    pub fn set(&mut self, position: usize) {
        if position >= N {
            todo!()
        }
        self.unchecked_set(position, true)
    }
    pub fn flip(&mut self, position: usize) {
        if position >= N {
            todo!()
        }
        self.unchecked_flip(position)
    }

    pub fn flip_all(&mut self) {
        self.base.do_flip()
    }

    pub fn unchecked_test(&self, pos: usize) -> bool {
        self.getword(pos) & CommonFns::maskbit(pos) != 0 as _WordT
    }
    pub fn check(&self, pos: usize) -> bool {
        self.unchecked_test(pos)
    }
    pub fn assign(&mut self, pos: usize, val: bool) {
        let mut ref_ = Reference::new(self, pos);
        unsafe { ref_.assign_value(val) };
    }
}
