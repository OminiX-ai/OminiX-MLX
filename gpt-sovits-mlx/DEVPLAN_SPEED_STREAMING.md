# Development Plan: Speed Factor & Streaming Support

**Status: Phase 1 & 2 COMPLETED (2025-02-01)**

## Overview

Add missing features to match Python primespeech interface:
1. **Speed Factor** - Control speech rate via feature interpolation
2. **Per-voice Speed Config** - Load speed_factor from voices.json
3. **Fragment Streaming** (future) - Incremental audio output

## Phase 1: Speed Factor Implementation (Priority: High)

### 1.1 Add MLX Linear Interpolation

**File:** `src/models/vits.rs`

**Current code (line ~1984):**
```rust
pub fn decode(
    &mut self,
    codes: &Array,
    text: &Array,
    refer: Option<&Array>,
    noise_scale: f32,
    _speed: f32,  // UNUSED!
) -> Result<Array, Exception>
```

**Target:** Implement 1D linear interpolation matching Python's `F.interpolate(y, mode="linear")`

**Algorithm:**
```
Input: tensor [batch, dim, seq_len], speed_factor
Output: tensor [batch, dim, new_len] where new_len = seq_len / speed

For each position i in [0, new_len):
    src_pos = i * speed  (float position in source)
    left = floor(src_pos)
    right = ceil(src_pos)
    weight = src_pos - left
    output[i] = source[left] * (1 - weight) + source[right] * weight
```

### 1.2 Integration Points

1. **vits.rs:decode()** - Apply interpolation after quantizer decode, before enc_p
2. **voice_clone.rs** - Already passes speed from config
3. **dora node config** - Already has SPEED_FACTOR env var

### 1.3 Test Cases

- speed=1.0 → output unchanged
- speed=1.1 → ~10% shorter audio (faster speech)
- speed=0.9 → ~11% longer audio (slower speech)
- Verify audio quality not degraded

## Phase 2: Per-Voice Speed Config (Priority: Medium)

### 2.1 Update voices.json Schema

**File:** `~/.dora/models/primespeech/voices.json`

**Add speed_factor field:**
```json
{
  "voices": {
    "doubao": {
      "ref_audio": "...",
      "ref_text": "...",
      "speed_factor": 1.1
    }
  }
}
```

### 2.2 Update Rust Config

**File:** `node-hub/dora-gpt-sovits-mlx/src/config.rs`

```rust
pub struct VoiceConfig {
    pub ref_audio: String,
    pub ref_text: String,
    pub vits_onnx: Option<String>,
    pub codes_path: Option<String>,
    pub speed_factor: Option<f32>,  // NEW
    pub aliases: Vec<String>,
}
```

**Priority:** env var > voice preset > default (1.0)

## Phase 3: Fragment Streaming (Priority: Low - Future)

### 3.1 Architecture Changes Required

Current flow (blocking):
```
synthesize(text) -> Vec<f32>  // Returns all audio at once
```

Streaming flow:
```
synthesize_stream(text) -> impl Iterator<Item=AudioChunk>
```

### 3.2 Implementation Approach

1. Split text into segments (reuse Python's cut algorithms or simple sentence split)
2. Process each segment through T2S → VITS
3. Yield audio chunk after each segment
4. Dora node emits multiple `audio` messages with `is_final` metadata

### 3.3 Complexity

- Requires significant refactoring of synthesis pipeline
- Need to handle cross-segment context for natural speech
- May need KV cache persistence across segments
- **Recommend deferring to future sprint**

## Implementation Order

| Task | Priority | Complexity | Time Est |
|------|----------|------------|----------|
| 1.1 MLX interpolation | High | Medium | 1-2h |
| 1.2 Integration in vits.rs | High | Low | 30min |
| 1.3 Unit tests | High | Low | 30min |
| 2.1 voices.json schema | Medium | Low | 15min |
| 2.2 Config parsing | Medium | Low | 30min |
| 3.x Streaming | Low | High | 4-8h |

## Success Criteria

1. `SPEED_FACTOR=1.1` produces ~10% faster speech
2. Audio quality comparable to Python implementation
3. Per-voice speed from voices.json works
4. All existing tests pass

## Files to Modify

1. `src/models/vits.rs` - Add interpolation
2. `src/audio/mod.rs` - Export interpolation helper (optional)
3. `node-hub/dora-gpt-sovits-mlx/src/config.rs` - Add speed_factor to VoiceConfig
4. `~/.dora/models/primespeech/voices.json` - Add speed_factor per voice
5. `tests/` - Add speed factor tests

---

## Completion Summary (2025-02-01)

### Completed Tasks

1. **MLX Linear Interpolation** ✅
   - Added `interpolate_linear()` function in `src/models/vits.rs`
   - Uses area-based sampling matching PyTorch's `F.interpolate(mode="linear")`
   - GPU-accelerated via MLX array operations

2. **VITS Decode Integration** ✅
   - Speed factor applied after quantizer decode, before enc_p forward
   - Formula: `new_len = seq_len / speed + 1`
   - Matches Python: `F.interpolate(y, size=int(y.shape[-1] / speed)+1, mode="linear")`

3. **Per-Voice Speed Config** ✅
   - Added `speed_factor: Option<f32>` to `VoiceConfig` struct
   - Config priority: env var > voice preset > default (1.0)
   - Updated `voices.json` with speed_factor for all voices:
     - Chinese voices: 1.1 (faster)
     - Marc (mixed): 1.0 (normal)

4. **Unit Tests** ✅
   - `test_interpolate_linear_same_size` - speed=1.0 unchanged
   - `test_interpolate_linear_downsample` - speed=2.0 half length
   - `test_interpolate_linear_upsample` - speed=0.5 double length
   - `test_interpolate_linear_speed_1_1` - speed=1.1 typical Chinese
   - `test_interpolate_linear_batch` - multi-batch/channel support

### Files Modified

| File | Changes |
|------|---------|
| `gpt-sovits-mlx/src/models/vits.rs` | Added `interpolate_linear()`, integrated in `decode()` |
| `dora-gpt-sovits-mlx/src/config.rs` | Added `speed_factor` to `VoiceConfig` |
| `dora-gpt-sovits-mlx/src/main.rs` | Added `use_gpu_mel: true` |
| `~/.dora/models/primespeech/voices.json` | Added `speed_factor` per voice |

### Remaining (Phase 3 - Future)

- Fragment streaming for real-time output
- Requires architectural changes to synthesis pipeline
- Recommend deferring to future sprint
