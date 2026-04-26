//! GPU memory monitoring and OOM-resilient evaluation.
//!
//! Provides safe wrappers around MLX memory APIs and an adaptive retry
//! mechanism for GPU operations that may fail under memory pressure.
//!
//! Inspired by Makepad's progressive buffer reservation strategy
//! (libs/llama/src/session.rs) which retries with incremental buffer
//! growth on Metal allocation failures.

/// GPU memory snapshot (all values in bytes).
#[derive(Debug, Clone, Copy)]
pub struct MemorySnapshot {
    /// Memory currently allocated and in use by MLX arrays.
    pub active: usize,
    /// Memory held in MLX's cache (freed arrays, available for reuse).
    pub cache: usize,
    /// Peak memory usage since process start or last reset.
    pub peak: usize,
}

impl MemorySnapshot {
    /// Total GPU memory currently held (active + cached).
    pub fn total(&self) -> usize {
        self.active + self.cache
    }
}

impl std::fmt::Display for MemorySnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "active={:.1}MB cache={:.1}MB peak={:.1}MB",
            self.active as f64 / 1e6,
            self.cache as f64 / 1e6,
            self.peak as f64 / 1e6,
        )
    }
}

/// Take a snapshot of current GPU memory usage.
pub fn memory_snapshot() -> MemorySnapshot {
    unsafe {
        let mut active: usize = 0;
        let mut cache: usize = 0;
        let mut peak: usize = 0;
        mlx_sys::mlx_get_active_memory(&mut active);
        mlx_sys::mlx_get_cache_memory(&mut cache);
        mlx_sys::mlx_get_peak_memory(&mut peak);
        MemorySnapshot { active, cache, peak }
    }
}

/// Clear MLX's GPU memory cache (frees unused cached allocations).
///
/// Safe to call at any point — only releases memory from completed
/// computation graphs, not in-flight operations.
pub fn clear_cache() {
    unsafe {
        mlx_sys::mlx_clear_cache();
    }
}

/// Set the MLX memory limit (maximum bytes MLX will allocate).
/// Returns the previous limit.
pub fn set_memory_limit(limit: usize) -> usize {
    unsafe {
        let mut prev: usize = 0;
        mlx_sys::mlx_set_memory_limit(&mut prev, limit);
        prev
    }
}

/// Set the MLX cache limit (maximum bytes MLX will hold in cache).
/// Returns the previous limit.
pub fn set_cache_limit(limit: usize) -> usize {
    unsafe {
        let mut prev: usize = 0;
        mlx_sys::mlx_set_cache_limit(&mut prev, limit);
        prev
    }
}

/// Evaluate an MLX array with OOM retry.
///
/// On failure, clears the GPU cache and retries up to `max_retries` times.
/// This handles transient memory pressure from accumulated cache entries
/// without requiring the caller to manage memory manually.
///
/// Returns `Ok(())` on success, or the last error if all retries fail.
pub fn eval_with_retry(
    arrays: &[&mlx_rs::Array],
    max_retries: usize,
) -> Result<(), mlx_rs::error::Exception> {
    let mut last_err = None;

    for attempt in 0..=max_retries {
        match mlx_rs::transforms::eval(arrays.iter().copied()) {
            Ok(()) => return Ok(()),
            Err(e) => {
                let msg = format!("{e}");
                let is_oom = msg.contains("out of memory")
                    || msg.contains("allocation failed")
                    || msg.contains("too small");

                if !is_oom || attempt == max_retries {
                    return Err(e);
                }

                // Log the retry attempt
                #[cfg(feature = "tracing")]
                tracing::warn!(
                    attempt = attempt + 1,
                    max_retries,
                    memory = %memory_snapshot(),
                    "OOM during eval, clearing cache and retrying"
                );

                clear_cache();
                last_err = Some(e);
            }
        }
    }

    Err(last_err.unwrap())
}

/// Guard that periodically clears GPU cache during long generation loops.
///
/// Tracks memory and clears cache when usage exceeds a threshold,
/// or every N steps — whichever comes first. Replaces the manual
/// `if step % 256 == 0 { mlx_clear_cache() }` pattern scattered
/// across model crates.
pub struct MemoryGuard {
    /// Clear cache every this many steps regardless of memory pressure.
    step_interval: usize,
    /// Clear cache when active memory exceeds this fraction of peak.
    pressure_threshold: f64,
    /// Current step counter.
    step: usize,
    /// Snapshot at last clear.
    last_clear: MemorySnapshot,
}

impl MemoryGuard {
    /// Create a new MemoryGuard.
    ///
    /// - `step_interval`: Force cache clear every N steps (e.g., 256).
    /// - `pressure_threshold`: Clear when `active / peak > threshold` (e.g., 0.9).
    pub fn new(step_interval: usize, pressure_threshold: f64) -> Self {
        Self {
            step_interval,
            pressure_threshold,
            step: 0,
            last_clear: memory_snapshot(),
        }
    }

    /// Default guard: clear every 256 steps or at 90% memory pressure.
    pub fn default_guard() -> Self {
        Self::new(256, 0.9)
    }

    /// Call once per generation step. Clears cache if needed.
    /// Returns true if cache was cleared.
    pub fn step(&mut self) -> bool {
        self.step += 1;

        // Always clear at step interval
        if self.step % self.step_interval == 0 {
            clear_cache();
            self.last_clear = memory_snapshot();
            return true;
        }

        // Check memory pressure
        let snap = memory_snapshot();
        if snap.peak > 0 {
            let pressure = snap.active as f64 / snap.peak as f64;
            if pressure > self.pressure_threshold {
                clear_cache();
                self.last_clear = memory_snapshot();
                return true;
            }
        }

        false
    }

    /// Current step count.
    pub fn current_step(&self) -> usize {
        self.step
    }

    /// Memory snapshot from last cache clear.
    pub fn last_snapshot(&self) -> MemorySnapshot {
        self.last_clear
    }

    /// Take a fresh memory snapshot.
    pub fn snapshot(&self) -> MemorySnapshot {
        memory_snapshot()
    }
}

/// Pre-flight memory check before starting inference.
///
/// Estimates whether there's enough GPU memory for the requested operation.
/// Returns `Ok(snapshot)` if there's likely enough memory, or
/// `Err(message)` with a description of the shortage.
///
/// This is advisory — actual memory needs depend on MLX's graph optimization.
/// But it catches obvious cases (e.g., loading a 13GB model on a 16GB device
/// with only 2GB free).
pub fn preflight_check(estimated_bytes: usize) -> Result<MemorySnapshot, String> {
    let snap = memory_snapshot();
    let available = if snap.cache > 0 {
        // Cache memory can be reclaimed
        snap.cache
    } else {
        0
    };

    // Conservative: if active + estimated > peak * 1.1, warn
    // (peak is our best proxy for "how much Metal let us allocate before")
    if snap.peak > 0 && snap.active + estimated_bytes > (snap.peak as f64 * 1.1) as usize + available {
        Err(format!(
            "Estimated {:.1}MB needed, but only ~{:.1}MB available \
             (active={:.1}MB, cache={:.1}MB, peak={:.1}MB). \
             Consider clearing cache or using a smaller model/batch.",
            estimated_bytes as f64 / 1e6,
            available as f64 / 1e6,
            snap.active as f64 / 1e6,
            snap.cache as f64 / 1e6,
            snap.peak as f64 / 1e6,
        ))
    } else {
        Ok(snap)
    }
}
