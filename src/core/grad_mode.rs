pub struct AutoGradMode {
    prev_mode: bool,
}

impl AutoGradMode {
    pub fn new(enabled: bool) -> Self {
        let self_ = Self {
            prev_mode: GradMode::is_enabled(),
        };
        GradMode::set_enabled(enabled);
        self_
    }
}

impl Drop for AutoGradMode {
    fn drop(&mut self) {
        GradMode::set_enabled(self.prev_mode)
    }
}

static mut GRADMODE_ENABLED: bool = true;
pub struct GradMode;

impl GradMode {
    pub fn is_enabled() -> bool {
        let t = unsafe { GRADMODE_ENABLED };
        // eprintln!("Gradmode_Status = {}", t);
        t
    }

    pub fn set_enabled(enabled: bool) {
        unsafe { GRADMODE_ENABLED = enabled }
    }
}

pub struct NoGradGuard {
    mode: AutoGradMode,
}

impl Default for NoGradGuard {
    fn default() -> Self {
        Self {
            mode: AutoGradMode::new(false),
        }
    }
}
