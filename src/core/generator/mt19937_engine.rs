const MERSENNE_STATE_N: usize = 624;
const MERSENNE_STATE_M: usize = 397;
const MATRIX_A: u32 = 0x9908b0df;
const UMASK: u32 = 0x80000000;
const LMASK: u32 = 0x7fffffff;

#[derive(Debug, Copy, Clone)]
pub struct MT19937DataPod {
    seed: u64,
    left: i32,
    seeded: bool,
    next: u32,
    state: [u32; MERSENNE_STATE_N as usize],
}

impl Default for MT19937DataPod {
    fn default() -> Self {
        Self {
            state: [0u32; MERSENNE_STATE_N as usize],
            seed: 0,
            left: 0,
            seeded: false,
            next: 0,
        }
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub struct MT19937Engine {
    data: MT19937DataPod,
}

impl MT19937Engine {
    #[inline]
    pub fn new(seed: Option<u64>) -> Self {
        let mut self_ = Self::default();
        match seed {
            Some(s) => self_.init_with_u32(s),
            _ => self_.init_with_u32(5489),
        }
        self_
    }

    #[inline]
    fn init_with_u32(&mut self, seed: u64) {
        self.data.seed = seed;
        self.data.seeded = true;
        self.data.state[0] = (seed & 0xffffffff) as u32;
        for j in 1..MERSENNE_STATE_N as u32 {
            self.data.state[j as usize] = 1812433253u32
                .overflowing_mul(
                    self.data.state[(j - 1) as usize] ^ (self.data.state[(j - 1) as usize] >> 30),
                )
                .0
                + j;
            // println!("{}",self.data.state[j as usize]);
            self.data.state[j as usize] &= 0xffffffff;
        }

        self.data.left = 1;
        self.data.next = 0;
    }

    pub fn call(&mut self) -> u32 {
        let mut y: u32;
        self.data.left -= 1;
        if self.data.left == 0 {
            self.next_state();
        }
        // y = 2925516573
        y = unsafe { *self.data.state.as_ptr().offset(self.data.next as isize) };
        self.data.next += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }

    #[inline]
    fn mix_bits(u: u32, v: u32) -> u32 {
        (u & UMASK) | (v & LMASK)
    }

    #[inline]
    fn twist(u: u32, v: u32) -> u32 {
        (Self::mix_bits(u, v) >> 1) ^ (if v & 1 > 0 { MATRIX_A } else { 0 })
    }

    #[inline]
    fn next_state(&mut self) {
        let mut p = self.data.state.as_mut_ptr();
        self.data.left = MERSENNE_STATE_N as i32;
        self.data.next = 0;

        // let mut count = 0;
        for _ in (1..(MERSENNE_STATE_N - MERSENNE_STATE_M + 1)).rev() {
            unsafe {
                // p: 0, count =102, j=125;
                *p = *p.offset(MERSENNE_STATE_M as isize) ^ Self::twist(*p.offset(0), *p.offset(1));
                p = p.offset(1);
            };
        }

        for _ in (1..MERSENNE_STATE_M).rev() {
            unsafe {
                *p = *p.offset(MERSENNE_STATE_M as isize - MERSENNE_STATE_N as isize)
                    ^ Self::twist(*p.offset(0), *p.offset(1));
                p = p.offset(1);
            }
        }

        unsafe {
            *p = *p.offset(MERSENNE_STATE_M as isize - MERSENNE_STATE_N as isize)
                ^ Self::twist(*p.offset(0), self.data.state[0]);
        }
    }

    #[inline]
    pub fn seed(&self) -> u64 {
        self.data.seed
    }
}
