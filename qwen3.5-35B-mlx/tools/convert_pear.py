"""Verify the published PEAR source and convert to MLX affine INT4/BF16 vision."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

REPO = "mit-han-lab/Qwen3.8-27B-PEAR"
REVISION = "65176e435e5fa80a059ebfafaf8215e6ad00a83e"


def preserve_processor_assets(source, destination):
    """Keep the upstream processor/template; mlx-vlm's custom serializer differs."""
    hashes = {}
    for name in (
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "LICENSE",
    ):
        shutil.copyfile(source / name, destination / name)
        hashes[name] = hashlib.sha256((destination / name).read_bytes()).hexdigest()
    return hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument(
        "--metadata", required=True, type=Path, help="HF revision API response with blobs=true"
    )
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    metadata = json.loads(args.metadata.read_text())
    if metadata["id"] != REPO or metadata["sha"] != REVISION:
        raise ValueError("Expected the pinned official PEAR checkpoint metadata")
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite an existing conversion")
    hashes = {}
    for item in metadata["siblings"]:
        if not item["rfilename"].endswith(".safetensors"):
            continue
        file = args.source / item["rfilename"]
        if file.stat().st_size != item["size"]:
            raise ValueError(f"Incomplete checkpoint shard: {file}")
        with file.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["lfs"]["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {file}")
        hashes[file.name] = digest
        print("Verified source", file.name, digest, flush=True)
    from mlx_vlm.convert import convert

    # mlx-vlm excludes the vision tower from its default quantization predicate.
    convert(
        str(args.source.resolve()),
        str(args.out.resolve()),
        quantize=True,
        q_group_size=64,
        q_bits=4,
        q_mode="affine",
        dtype="bfloat16",
        trust_remote_code=False,
    )
    assets = preserve_processor_assets(args.source, args.out)
    import mlx.core as mx

    index = json.loads((args.out / "model.safetensors.index.json").read_text())["weight_map"]
    derived = {}
    vision_count = 0
    for name in sorted(set(index.values())):
        file = args.out / name
        weights = mx.load(str(file))
        for key, value in weights.items():
            if key.startswith("vision_tower."):
                if value.dtype != mx.bfloat16:
                    raise ValueError(f"Expected unquantized BF16 vision: {key} {value.dtype}")
                vision_count += 1
        del weights
        with file.open("rb") as stream:
            derived[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    if vision_count != 333:
        raise ValueError(f"Expected all 333 vision tensors, got {vision_count}")
    report = {
        "source_repo": REPO,
        "source_revision": REVISION,
        "source_sha256": hashes,
        "converted_sha256": derived,
        "language_quantization": {"mode": "affine", "bits": 4, "group_size": 64},
        "vision_dtype": "bfloat16",
        "vision_tensors": vision_count,
        "processor_assets_sha256": assets,
        "conversion_packages": {
            name: importlib.metadata.version(name) for name in ("mlx", "mlx-vlm", "transformers")
        },
    }
    (args.out / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print("CONVERSION_COMPLETE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
