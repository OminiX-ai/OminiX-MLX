use anyhow::{ensure, Context, Result};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct Region {
    pub text: String,
    /// Source-image pixel coordinates, in model output order.
    pub polygon: [[f64; 2]; 4],
}

pub(crate) fn parse(raw: &str, width: u32, height: u32, truncated: bool) -> Result<Vec<Region>> {
    if raw.contains("<|TEXT_START|>") || raw.contains("<|LOC_BEGIN|>") {
        let texts = tagged(raw, "<|TEXT_START|>", "<|TEXT_END|>", truncated)?;
        let boxes = tagged(raw, "<|LOC_BEGIN|>", "<|LOC_END|>", truncated)?;
        ensure!(
            texts.len() == boxes.len() || truncated,
            "unpaired tagged spotting output"
        );
        let mut regions = Vec::new();
        for (label, locations) in texts.into_iter().zip(boxes) {
            let pair = format!("{label}{locations}");
            let mut parsed = parse_plain(&pair, width, height, false)?;
            ensure!(parsed.len() == 1, "expected one polygon per tagged region");
            regions.append(&mut parsed);
        }
        return Ok(regions);
    }
    parse_plain(raw, width, height, truncated)
}

fn tagged<'a>(mut rest: &'a str, begin: &str, end: &str, truncated: bool) -> Result<Vec<&'a str>> {
    let mut values = Vec::new();
    while let Some((_, body)) = rest.split_once(begin) {
        let Some((value, remaining)) = body.split_once(end) else {
            ensure!(truncated, "unterminated {begin} group");
            break;
        };
        values.push(value);
        rest = remaining;
    }
    Ok(values)
}

fn parse_plain(raw: &str, width: u32, height: u32, truncated: bool) -> Result<Vec<Region>> {
    let mut regions = Vec::new();
    let mut rest = raw.trim();
    while !rest.is_empty() {
        let Some((label, mut coords)) = rest.split_once("<|LOC_") else {
            ensure!(truncated, "text with no polygon at end of spotting output");
            break;
        };
        let label = label.trim();
        ensure!(!label.is_empty(), "polygon without text");
        let mut polygon = [[0.0; 2]; 4];
        for i in 0..8 {
            if i > 0 {
                let Some(next) = coords.trim_start().strip_prefix("<|LOC_") else {
                    if truncated && coords.trim().is_empty() {
                        return Ok(regions);
                    }
                    anyhow::bail!("incomplete polygon for {label}");
                };
                coords = next;
            }
            let Some((n, tail)) = coords.split_once("|>") else {
                if truncated {
                    return Ok(regions);
                }
                anyhow::bail!("unterminated coordinate for {label}");
            };
            let n: u16 = n.parse().context("invalid location token")?;
            ensure!(
                n <= 1000,
                "location coordinate {n} outside normalized range"
            );
            let extent = if i % 2 == 0 { width } else { height };
            polygon[i / 2][i % 2] = f64::from(n) * f64::from(extent) / 1000.0;
            coords = tail;
        }
        regions.push(Region {
            text: label.to_owned(),
            polygon,
        });
        rest = coords.trim();
    }
    Ok(regions)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_unicode_and_scales_each_axis() {
        let input="文字 A<|LOC_50|><|LOC_20|><|LOC_800|><|LOC_30|><|LOC_800|><|LOC_990|><|LOC_50|><|LOC_980|>";
        let r = parse(input, 200, 100, false).unwrap();
        assert_eq!(r[0].text, "文字 A");
        assert_eq!(
            r[0].polygon,
            [[10., 2.], [160., 3.], [160., 99.], [10., 98.]]
        );
    }
    #[test]
    fn truncated_polygon_is_never_presented_as_complete() {
        assert!(parse("A<|LOC_20|>", 100, 100, false).is_err());
        assert!(parse("A<|LOC_20|>", 100, 100, true).unwrap().is_empty());
    }

    #[test]
    fn tagged_positions_and_transcriptions_are_paired() {
        let coords =
            "<|LOC_0|><|LOC_0|><|LOC_1000|><|LOC_0|><|LOC_1000|><|LOC_1000|><|LOC_0|><|LOC_1000|>";
        let tagged = format!("<|LOC_BEGIN|>{coords}<|LOC_END|><|TEXT_START|>甲<|TEXT_END|>");
        assert_eq!(
            parse(&tagged, 200, 100, false).unwrap()[0].polygon[2],
            [200., 100.]
        );
    }
}
