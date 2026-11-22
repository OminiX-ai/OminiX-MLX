use anyhow::{ensure, Result};
use image::{
    imageops::{resize, FilterType},
    DynamicImage, GenericImageView,
};
use mlx_rs::Array;

pub(crate) struct Input {
    pub pixels: Array,
    pub grid: [i32; 2],
}

pub(crate) fn prepare(image: &DynamicImage, patch: i32, merge: i32) -> Result<Input> {
    let (w, h) = image.dimensions();
    ensure!(w > 0 && h > 0, "image is empty");
    ensure!(
        w.max(h) as f64 / w.min(h) as f64 <= 200.0,
        "aspect ratio exceeds 200:1"
    );
    let rgb = image.to_rgb8();
    let rgb = if w < 1500 && h < 1500 {
        resize(&rgb, 2 * w, 2 * h, FilterType::Lanczos3)
    } else {
        rgb
    };
    let factor = f64::from(patch * merge);
    let (mut w, mut h) = (f64::from(rgb.width()), f64::from(rgb.height()));
    if h < factor {
        w = (w * factor / h).round_ties_even();
        h = factor;
    }
    if w < factor {
        h = (h * factor / w).round_ties_even();
        w = factor;
    }
    let mut cols = (w / factor).round_ties_even();
    let mut rows = (h / factor).round_ties_even();
    let min = 112896.0;
    let max = 2048.0 * 28.0 * 28.0;
    let rounded = cols * rows * factor * factor;
    if rounded > max {
        let scale = (w * h / max).sqrt() * factor;
        cols = (w / scale).floor();
        rows = (h / scale).floor();
    } else if rounded < min {
        let scale = (min / (w * h)).sqrt() / factor;
        cols = (w * scale).ceil();
        rows = (h * scale).ceil();
    }
    let width = (cols * factor) as u32;
    let height = (rows * factor) as u32;
    ensure!(width > 0 && height > 0, "invalid resized image");
    let rgb = resize(&rgb, width, height, FilterType::CatmullRom);
    let data: Vec<f32> = rgb
        .as_raw()
        .iter()
        .map(|&x| (x as f32 - 127.5) / 127.5)
        .collect();
    Ok(Input {
        pixels: Array::from_slice(&data, &[1, height as i32, width as i32, 3]),
        grid: [height as i32 / patch, width as i32 / patch],
    })
}
