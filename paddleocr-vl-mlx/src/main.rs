use anyhow::{Context, Result};
use clap::Parser;
use mlx_rs::Device;
use paddleocr_vl_mlx::PaddleOcr;
use std::{
    io::{self, Write},
    path::PathBuf,
    time::Instant,
};

/// Spot text lines and quadrilaterals on one or more images using a resident PaddleOCR-VL model.
#[derive(Parser)]
struct Args {
    /// Directory containing the official BF16 model.safetensors, config.json and tokenizer.json.
    #[arg(long)]
    model_dir: PathBuf,
    /// PNG, JPEG, or WebP. Supply several paths to reuse the loaded model.
    #[arg(long,required=true,num_args=1..)]
    image: Vec<PathBuf>,
    #[arg(long, default_value_t = 3000)]
    max_tokens: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    Device::set_default(&Device::gpu());
    let t = Instant::now();
    let model = PaddleOcr::load(&args.model_dir)?;
    eprintln!("Model initialized in {:.3}s", t.elapsed().as_secs_f64());
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for path in &args.image {
        let start = Instant::now();
        let image = image::open(path).with_context(|| format!("opening {}", path.display()))?;
        let result = model
            .spot(&image, args.max_tokens)
            .with_context(|| format!("spotting {}", path.display()))?;
        let seconds = start.elapsed().as_secs_f64();
        let record = serde_json::json!({"image":path,"seconds":seconds,"spotting":result});
        serde_json::to_writer(&mut stdout, &record)?;
        writeln!(stdout)?;
        stdout.flush()?;
    }
    Ok(())
}
