//! Learning rate schedulers

use std::f32::consts::PI;

/// Trait for learning rate schedulers
pub trait LRScheduler {
    /// Get current learning rate
    fn get_lr(&self) -> f32;

    /// Step the scheduler (call after each optimization step)
    fn step(&mut self);

    /// Get current step number
    fn current_step(&self) -> usize;

    /// Reset scheduler to initial state
    fn reset(&mut self);
}

/// Cosine annealing scheduler with optional warmup
#[derive(Debug, Clone)]
pub struct CosineScheduler {
    /// Base learning rate
    base_lr: f32,
    /// Minimum learning rate
    min_lr: f32,
    /// Number of warmup steps
    warmup_steps: usize,
    /// Total number of steps
    total_steps: usize,
    /// Current step
    current_step: usize,
}

impl CosineScheduler {
    /// Create a new cosine scheduler
    pub fn new(base_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            base_lr,
            min_lr: 0.0,
            warmup_steps,
            total_steps,
            current_step: 0,
        }
    }

    /// Set minimum learning rate
    pub fn with_min_lr(mut self, min_lr: f32) -> Self {
        self.min_lr = min_lr;
        self
    }

    /// Set current step (useful for resuming training)
    pub fn set_step(&mut self, step: usize) {
        self.current_step = step;
    }
}

impl LRScheduler for CosineScheduler {
    fn get_lr(&self) -> f32 {
        if self.current_step < self.warmup_steps {
            // Linear warmup
            let warmup_ratio = self.current_step as f32 / self.warmup_steps as f32;
            self.base_lr * warmup_ratio
        } else {
            // Cosine annealing
            let progress = (self.current_step - self.warmup_steps) as f32
                / (self.total_steps - self.warmup_steps).max(1) as f32;
            let progress = progress.min(1.0);

            // Cosine decay from base_lr to min_lr
            let decay = 0.5 * (1.0 + (PI * progress).cos());
            self.min_lr + (self.base_lr - self.min_lr) * decay
        }
    }

    fn step(&mut self) {
        self.current_step += 1;
    }

    fn current_step(&self) -> usize {
        self.current_step
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}

/// Linear warmup scheduler (constant LR after warmup)
#[derive(Debug, Clone)]
pub struct WarmupScheduler {
    /// Base learning rate
    base_lr: f32,
    /// Number of warmup steps
    warmup_steps: usize,
    /// Current step
    current_step: usize,
}

impl WarmupScheduler {
    /// Create a new warmup scheduler
    pub fn new(base_lr: f32, warmup_steps: usize) -> Self {
        Self {
            base_lr,
            warmup_steps,
            current_step: 0,
        }
    }
}

impl LRScheduler for WarmupScheduler {
    fn get_lr(&self) -> f32 {
        if self.current_step < self.warmup_steps {
            let warmup_ratio = (self.current_step + 1) as f32 / self.warmup_steps as f32;
            self.base_lr * warmup_ratio
        } else {
            self.base_lr
        }
    }

    fn step(&mut self) {
        self.current_step += 1;
    }

    fn current_step(&self) -> usize {
        self.current_step
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}

/// Linear decay scheduler with optional warmup
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct LinearScheduler {
    /// Base learning rate
    base_lr: f32,
    /// End learning rate
    end_lr: f32,
    /// Number of warmup steps
    warmup_steps: usize,
    /// Total number of steps
    total_steps: usize,
    /// Current step
    current_step: usize,
}

#[allow(dead_code)]
impl LinearScheduler {
    /// Create a new linear scheduler
    pub fn new(base_lr: f32, end_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            base_lr,
            end_lr,
            warmup_steps,
            total_steps,
            current_step: 0,
        }
    }
}

impl LRScheduler for LinearScheduler {
    fn get_lr(&self) -> f32 {
        if self.current_step < self.warmup_steps {
            // Linear warmup
            let warmup_ratio = self.current_step as f32 / self.warmup_steps as f32;
            self.base_lr * warmup_ratio
        } else {
            // Linear decay
            let decay_steps = self.total_steps - self.warmup_steps;
            let progress = (self.current_step - self.warmup_steps) as f32 / decay_steps.max(1) as f32;
            let progress = progress.min(1.0);
            self.base_lr + (self.end_lr - self.base_lr) * progress
        }
    }

    fn step(&mut self) {
        self.current_step += 1;
    }

    fn current_step(&self) -> usize {
        self.current_step
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_scheduler_warmup() {
        let mut scheduler = CosineScheduler::new(1e-4, 100, 1000);

        // During warmup, LR should increase linearly
        assert_eq!(scheduler.get_lr(), 0.0);

        for _ in 0..50 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 5e-5).abs() < 1e-7);

        for _ in 0..50 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 1e-4).abs() < 1e-7);
    }

    #[test]
    fn test_cosine_scheduler_decay() {
        let mut scheduler = CosineScheduler::new(1e-4, 0, 1000);

        // At step 0, should be at base_lr
        assert!((scheduler.get_lr() - 1e-4).abs() < 1e-7);

        // At step 500 (halfway), should be at base_lr * 0.5
        for _ in 0..500 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 5e-5).abs() < 1e-6);

        // At step 1000, should be at 0
        for _ in 0..500 {
            scheduler.step();
        }
        assert!(scheduler.get_lr() < 1e-7);
    }

    #[test]
    fn test_warmup_scheduler() {
        let mut scheduler = WarmupScheduler::new(1e-4, 100);

        // During warmup
        assert!((scheduler.get_lr() - 1e-6).abs() < 1e-8);

        for _ in 0..100 {
            scheduler.step();
        }
        // After warmup, should be at base_lr
        assert!((scheduler.get_lr() - 1e-4).abs() < 1e-7);

        // Should stay constant after warmup
        for _ in 0..100 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 1e-4).abs() < 1e-7);
    }

    #[test]
    fn test_linear_scheduler() {
        let mut scheduler = LinearScheduler::new(1e-4, 1e-5, 0, 1000);

        // At start
        assert!((scheduler.get_lr() - 1e-4).abs() < 1e-7);

        // Halfway
        for _ in 0..500 {
            scheduler.step();
        }
        let expected = 1e-4 + (1e-5 - 1e-4) * 0.5;
        assert!((scheduler.get_lr() - expected).abs() < 1e-7);

        // At end
        for _ in 0..500 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 1e-5).abs() < 1e-7);
    }
}
