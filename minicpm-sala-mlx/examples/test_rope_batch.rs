use mlx_rs::{builder::Builder, nn, module::Module, ops::indexing::IndexOp, transforms::eval};

fn main() -> anyhow::Result<()> {
    // Create identical data for both batch elements
    let single = mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[1, 32, 1, 128], None)?;
    eval([&single])?;

    // Repeat to create [2, 32, 1, 128] with identical batch elements
    let batched = mlx_rs::ops::concatenate_axis(&[&single, &single], 0)?;
    eval([&batched])?;

    // Verify input is identical
    let in0 = batched.index((0, 0, 0, ..));
    let in1 = batched.index((1, 0, 0, ..));
    eval([&in0, &in1])?;
    let diff = in0.subtract(&in1)?;
    let abs_diff = mlx_rs::ops::abs(&diff)?;
    let max_diff = mlx_rs::ops::max_axis(&abs_diff, 0, None)?;
    eval([&max_diff])?;
    println!("Input max |b0-b1| = {:.10}", max_diff.item::<f32>());

    // Apply RoPE directly using fast::rope
    println!("Using mlx_rs::fast::rope directly");

    let output = mlx_rs::fast::rope(&batched, 128, false, 10000.0, 1.0, 8, None)?;
    eval([&output])?;
    let out0 = output.index((0, 0, 0, ..));
    let out1 = output.index((1, 0, 0, ..));
    eval([&out0, &out1])?;
    let diff = out0.subtract(&out1)?;
    let abs_diff = mlx_rs::ops::abs(&diff)?;
    let max_diff = mlx_rs::ops::max_axis(&abs_diff, 0, None)?;
    eval([&max_diff])?;
    println!("fast::rope 4D offset=8 max |b0-b1| = {:.10}", max_diff.item::<f32>());

    let output0 = mlx_rs::fast::rope(&batched, 128, false, 10000.0, 1.0, 0, None)?;
    eval([&output0])?;
    let out0 = output0.index((0, 0, 0, ..));
    let out1 = output0.index((1, 0, 0, ..));
    eval([&out0, &out1])?;
    let diff = out0.subtract(&out1)?;
    let abs_diff = mlx_rs::ops::abs(&diff)?;
    let max_diff = mlx_rs::ops::max_axis(&abs_diff, 0, None)?;
    eval([&max_diff])?;
    println!("fast::rope 4D offset=0 max |b0-b1| = {:.10}", max_diff.item::<f32>());

    // Use nn::Rope module
    use mlx_rs_core::utils::initialize_rope;
    let mut rope = initialize_rope(128, 10000.0, false, &None, 524288)?;

    let input = nn::RopeInputBuilder::new(&batched)
        .offset(8)
        .build()?;
    let output = rope.forward(input)?;
    eval([&output])?;

    // Compare batch elements
    let out0 = output.index((0, 0, 0, ..));
    let out1 = output.index((1, 0, 0, ..));
    eval([&out0, &out1])?;
    let diff = out0.subtract(&out1)?;
    let abs_diff = mlx_rs::ops::abs(&diff)?;
    let max_diff = mlx_rs::ops::max_axis(&abs_diff, 0, None)?;
    eval([&max_diff])?;
    println!("RoPE output max |b0-b1| = {:.10}", max_diff.item::<f32>());

    // Also test with offset=0
    let input2 = nn::RopeInputBuilder::new(&batched)
        .offset(0)
        .build()?;
    let output2 = rope.forward(input2)?;
    eval([&output2])?;
    let out0 = output2.index((0, 0, 0, ..));
    let out1 = output2.index((1, 0, 0, ..));
    eval([&out0, &out1])?;
    let diff = out0.subtract(&out1)?;
    let abs_diff = mlx_rs::ops::abs(&diff)?;
    let max_diff = mlx_rs::ops::max_axis(&abs_diff, 0, None)?;
    eval([&max_diff])?;
    println!("RoPE offset=0 max |b0-b1| = {:.10}", max_diff.item::<f32>());

    // Workaround: reshape [B, H, L, D] to [1, B*H, L, D] before RoPE
    println!("\n--- Workaround: reshape to [1, B*H, L, D] ---");
    let shape = batched.shape().to_vec();
    let b = shape[0]; let h = shape[1]; let l = shape[2]; let d = shape[3];
    let reshaped = batched.reshape(&[1, b * h, l, d])?;
    let output_r = mlx_rs::fast::rope(&reshaped, 128, false, 10000.0, 1.0, 8, None)?;
    let output_fix = output_r.reshape(&[b, h, l, d])?;
    eval([&output_fix])?;
    let out0 = output_fix.index((0, 0, 0, ..));
    let out1 = output_fix.index((1, 0, 0, ..));
    eval([&out0, &out1])?;
    let diff = out0.subtract(&out1)?;
    let abs_diff = mlx_rs::ops::abs(&diff)?;
    let max_diff = mlx_rs::ops::max_axis(&abs_diff, 0, None)?;
    eval([&max_diff])?;
    println!("Workaround [1,B*H,L,D] offset=8 max |b0-b1| = {:.10}", max_diff.item::<f32>());

    // Verify correctness: compare workaround output for batch 0 vs original single-batch rope
    let single_rope = mlx_rs::fast::rope(&single, 128, false, 10000.0, 1.0, 8, None)?;
    eval([&single_rope])?;
    let ref_out = single_rope.index((0, 0, 0, ..));
    let fix_out = output_fix.index((0, 0, 0, ..));
    eval([&ref_out, &fix_out])?;
    let diff = ref_out.subtract(&fix_out)?;
    let abs_diff = mlx_rs::ops::abs(&diff)?;
    let max_diff = mlx_rs::ops::max_axis(&abs_diff, 0, None)?;
    eval([&max_diff])?;
    println!("Workaround vs single-batch reference max diff = {:.10}", max_diff.item::<f32>());

    // Test prefill shape [2, 32, 8, 128]
    println!("\n--- Prefill shape test [2, 32, 8, 128] ---");
    let single_pf = mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[1, 32, 8, 128], None)?;
    eval([&single_pf])?;
    let batched_pf = mlx_rs::ops::concatenate_axis(&[&single_pf, &single_pf], 0)?;
    eval([&batched_pf])?;
    let output_pf = mlx_rs::fast::rope(&batched_pf, 128, false, 10000.0, 1.0, 0, None)?;
    eval([&output_pf])?;
    let pf0 = output_pf.index((0, 0, .., ..)); // [8, 128]
    let pf1 = output_pf.index((1, 0, .., ..)); // [8, 128]
    eval([&pf0, &pf1])?;
    let diff = pf0.subtract(&pf1)?;
    let abs_diff = mlx_rs::ops::abs(&diff)?;
    let md1 = mlx_rs::ops::max_axis(&abs_diff, 0, None)?;
    let md2 = mlx_rs::ops::max_axis(&md1, 0, None)?;
    eval([&md2])?;
    println!("Prefill [2,32,8,128] offset=0 max |b0-b1| = {:.10}", md2.item::<f32>());

    Ok(())
}
