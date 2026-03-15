# Releasing OminiX-MLX

## Prerequisites

```bash
cargo install cargo-release git-cliff
```

## Release the workspace (mlx-rs + all model crates)

```bash
# Preview what will happen
cargo release patch --dry-run

# Release (bumps workspace version, generates changelog, commits, tags, pushes)
cargo release patch             # 0.25.2 → 0.25.3
cargo release minor             # 0.25.2 → 0.26.0
cargo release major             # 0.25.2 → 1.0.0
```

This bumps all crates that use `version.workspace = true`. Excluded crates (independent versions):
- `mlx-sys` (tracks mlx-c, currently 0.2.0)
- `ominix-api` (independent, currently 1.0.0)
- `mlx-lm-utils` (independent, currently 0.0.1)
- `mlx-rs-core` (independent, currently 0.1.0)

## Release ominix-api member separately

```bash
cargo release patch -p ominix-api --tag-prefix ominix-api-v
```

This tags as `ominix-api-v1.0.1` and triggers `.github/workflows/release-ominix-api.yml`, which builds the binary and creates a GitHub release.

## What happens

1. `cargo release patch` bumps `workspace.package.version` in root `Cargo.toml`
2. Runs `git-cliff` to update `CHANGELOG.md`
3. Commits: `release: v0.25.3`
4. Tags: `v0.25.3`
5. Pushes commit + tag to `ominix/main`
6. Tag push triggers `.github/workflows/release.yml`:
   - Runs full test suite
   - Creates GitHub release with changelog

## Commit message format

Use [conventional commits](https://www.conventionalcommits.org/) for automatic changelog grouping:

```
feat: add streaming support     → Features
fix: handle edge case           → Bug Fixes
perf: optimize inference        → Performance
refactor: simplify model loader → Refactor
docs: update API docs           → Documentation
ci: fix workflow                → CI
```

## Workflows

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `release.yml` | `v*` tags | Tests + GitHub release for workspace |
| `release-ominix-api.yml` | `ominix-api-v*` tags | Builds binary + GitHub release |
| `validate.yml` | Push/PR | CI: fmt, clippy, tests |
| `publish-docs.yml` | Push to main | Deploys API docs to GitHub Pages |

## Notes

- `publish = false` — crates are not published to crates.io
- Always `--dry-run` first to verify
- The ominix-api release workflow generates scoped changelog (only `ominix-api/**` changes)
