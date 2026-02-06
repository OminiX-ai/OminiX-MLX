extern crate cmake;

use std::{env, fs, path::PathBuf, process::Command};

/// Version tag for the GitHub Release containing pre-built artifacts.
const PREBUILT_RELEASE_TAG: &str = "mlx-prebuilt-v0.1.0";

/// GitHub repo for downloading pre-built artifacts.
const GITHUB_REPO: &str = "OminiX-ai/OminiX-MLX";

// ─── Detection ───────────────────────────────────────────────────

fn explicit_prebuilt_path() -> Option<PathBuf> {
    env::var("MLX_PREBUILT_PATH").ok().map(PathBuf::from)
}

fn metal_compiler_available() -> bool {
    Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "--version"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

// ─── Download ────────────────────────────────────────────────────

fn download_prebuilt() -> PathBuf {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let prebuilt_dir = out_dir.join("mlx-prebuilt");

    // If already downloaded (incremental build), reuse
    if prebuilt_dir.join("libmlx.a").exists()
        && prebuilt_dir.join("libmlxc.a").exists()
        && prebuilt_dir.join("mlx.metallib").exists()
    {
        println!(
            "cargo:warning=Reusing cached pre-built MLX from: {}",
            prebuilt_dir.display()
        );
        return prebuilt_dir;
    }

    fs::create_dir_all(&prebuilt_dir).expect("Failed to create prebuilt dir");

    let arch = if cfg!(target_arch = "aarch64") {
        "arm64"
    } else {
        "x86_64"
    };
    let asset_name = format!(
        "{}-macos-{}.tar.gz",
        PREBUILT_RELEASE_TAG, arch
    );
    let url = format!(
        "https://github.com/{}/releases/download/{}/{}",
        GITHUB_REPO, PREBUILT_RELEASE_TAG, asset_name
    );

    println!("cargo:warning=Downloading pre-built MLX from: {}", url);

    let tarball = out_dir.join(&asset_name);

    // Download using curl (available on all macOS systems)
    let status = Command::new("curl")
        .args(["-L", "-f", "--progress-bar", "-o"])
        .arg(&tarball)
        .arg(&url)
        .status()
        .expect("Failed to run curl");

    if !status.success() {
        panic!(
            "Failed to download pre-built MLX artifacts from {}.\n\
             Either install Xcode to build from source, or set MLX_PREBUILT_PATH \
             to a directory containing libmlx.a, libmlxc.a, libgguflib.a, and mlx.metallib.",
            url
        );
    }

    // Extract
    let status = Command::new("tar")
        .args(["-xzf"])
        .arg(&tarball)
        .arg("-C")
        .arg(&out_dir)
        .status()
        .expect("Failed to run tar");

    if !status.success() {
        panic!("Failed to extract pre-built MLX tarball");
    }

    // The tarball extracts to a directory matching the asset name (minus .tar.gz)
    let extracted_dir = out_dir.join(asset_name.trim_end_matches(".tar.gz"));

    // Move contents to our canonical prebuilt_dir
    if extracted_dir.exists() {
        for entry in fs::read_dir(&extracted_dir).expect("Failed to read extracted dir") {
            let entry = entry.unwrap();
            let dest = prebuilt_dir.join(entry.file_name());
            if let Err(_) = fs::rename(entry.path(), &dest) {
                fs::copy(entry.path(), &dest).expect("Failed to copy artifact");
            }
        }
        fs::remove_dir_all(&extracted_dir).ok();
    }

    fs::remove_file(&tarball).ok();

    println!(
        "cargo:warning=Pre-built MLX extracted to: {}",
        prebuilt_dir.display()
    );
    prebuilt_dir
}

// ─── Metallib placement ─────────────────────────────────────────

fn copy_metallib_to_target_dir(metallib_src: &PathBuf) {
    if !metallib_src.exists() {
        return;
    }

    // OUT_DIR is like: <workspace>/target/release/build/mlx-sys-<hash>/out
    // We want: <workspace>/target/release/
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mut dir = out_dir.as_path();

    while let Some(parent) = dir.parent() {
        if dir.file_name().map(|f| f == "build").unwrap_or(false) {
            let target_profile_dir = parent;
            let dest = target_profile_dir.join("mlx.metallib");
            let src_len = fs::metadata(metallib_src).map(|m| m.len()).unwrap_or(0);
            let dst_len = fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
            if !dest.exists() || src_len != dst_len {
                println!(
                    "cargo:warning=Copying mlx.metallib ({} MB) to {}",
                    src_len / 1_000_000,
                    dest.display()
                );
                fs::copy(metallib_src, &dest)
                    .expect("Failed to copy mlx.metallib to target directory");
            }
            return;
        }
        dir = parent;
    }

    println!("cargo:warning=Could not determine target directory for mlx.metallib placement");
}

// ─── Linking ─────────────────────────────────────────────────────

fn link_prebuilt(prebuilt_dir: &PathBuf) {
    println!(
        "cargo:rustc-link-search=native={}",
        prebuilt_dir.display()
    );
    println!("cargo:rustc-link-lib=static=mlx");
    println!("cargo:rustc-link-lib=static=mlxc");

    // gguflib may or may not be a separate archive
    if prebuilt_dir.join("libgguflib.a").exists() {
        println!("cargo:rustc-link-lib=static=gguflib");
    }

    let metallib = prebuilt_dir.join("mlx.metallib");
    copy_metallib_to_target_dir(&metallib);
}

/// Patch the MLX source files to work around macOS Tahoe beta issues
fn patch_metal_version(out_dir: &PathBuf) {
    // Patch device.cpp to force Metal 3.2
    let device_cpp = out_dir.join("build/_deps/mlx-src/mlx/backend/metal/device.cpp");
    if device_cpp.exists() {
        if let Ok(content) = fs::read_to_string(&device_cpp) {
            if !content.contains("// PATCHED: Force Metal 3.2") {
                let old_code = r#"auto get_metal_version() {
  auto get_metal_version_ = []() {
    if (__builtin_available(macOS 26, iOS 26, tvOS 26, visionOS 26, *)) {
      return MTL::LanguageVersion4_0;
    } else if (__builtin_available(macOS 15, iOS 18, tvOS 18, visionOS 2, *)) {
      return MTL::LanguageVersion3_2;
    } else {
      return MTL::LanguageVersion3_1;
    }
  };
  static auto metal_version_ = get_metal_version_();
  return metal_version_;
}"#;

                let new_code = r#"// PATCHED: Force Metal 3.2 to work around Xcode beta Metal 4.0 issues
auto get_metal_version() {
  auto get_metal_version_ = []() {
    // Force Metal 3.2 - Metal 4.0 not supported in current Xcode beta
    if (__builtin_available(macOS 15, iOS 18, tvOS 18, visionOS 2, *)) {
      return MTL::LanguageVersion3_2;
    } else {
      return MTL::LanguageVersion3_1;
    }
  };
  static auto metal_version_ = get_metal_version_();
  return metal_version_;
}"#;

                if content.contains(old_code) {
                    let patched = content.replace(old_code, new_code);
                    if fs::write(&device_cpp, patched).is_ok() {
                        println!("cargo:warning=Patched MLX device.cpp to force Metal 3.2");
                    }
                }
            }
        }
    }

    // Patch device.h to disable NAX (uses __builtin_available for macOS 26.2)
    let device_h = out_dir.join("build/_deps/mlx-src/mlx/backend/metal/device.h");
    if device_h.exists() {
        if let Ok(content) = fs::read_to_string(&device_h) {
            if !content.contains("// PATCHED: Disable NAX") {
                let old_code = r#"inline bool is_nax_available() {
  auto _check_nax = []() {
    bool can_use_nax = false;
    if (__builtin_available(
            macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *)) {
      can_use_nax = true;
    }
    can_use_nax &=
        metal::device(mlx::core::Device::gpu).get_architecture_gen() >= 17;
    return can_use_nax;
  };
  static bool is_nax_available_ = _check_nax();
  return is_nax_available_;
}"#;

                let new_code = r#"// PATCHED: Disable NAX - __builtin_available for macOS 26.2 causes link errors
inline bool is_nax_available() {
  // NAX is not available on current Xcode beta
  return false;
}"#;

                if content.contains(old_code) {
                    let patched = content.replace(old_code, new_code);
                    if fs::write(&device_h, patched).is_ok() {
                        println!("cargo:warning=Patched MLX device.h to disable NAX");
                    }
                }
            }
        }
    }
}

fn build_from_source() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let build_dir = out_dir.join("build");

    fs::create_dir_all(&build_dir).ok();

    let src_dir = PathBuf::from("src/mlx-c").canonicalize().unwrap();

    let mut cmake_args = vec![
        format!("-S{}", src_dir.display()),
        format!("-B{}", build_dir.display()),
        "-DCMAKE_INSTALL_PREFIX=.".to_string(),
    ];

    #[cfg(debug_assertions)]
    cmake_args.push("-DCMAKE_BUILD_TYPE=Debug".to_string());

    #[cfg(not(debug_assertions))]
    cmake_args.push("-DCMAKE_BUILD_TYPE=Release".to_string());

    #[cfg(feature = "metal")]
    cmake_args.push("-DMLX_BUILD_METAL=ON".to_string());

    #[cfg(not(feature = "metal"))]
    cmake_args.push("-DMLX_BUILD_METAL=OFF".to_string());

    #[cfg(feature = "accelerate")]
    cmake_args.push("-DMLX_BUILD_ACCELERATE=ON".to_string());

    #[cfg(not(feature = "accelerate"))]
    cmake_args.push("-DMLX_BUILD_ACCELERATE=OFF".to_string());

    cmake_args.push("-DCMAKE_METAL_COMPILER=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/metal".to_string());

    cmake_args.push("-DCMAKE_OSX_DEPLOYMENT_TARGET=15.0".to_string());

    std::env::set_var("MACOSX_DEPLOYMENT_TARGET", "15.0");

    let status = Command::new("cmake")
        .args(&cmake_args)
        .status()
        .expect("Failed to run cmake configure");

    if !status.success() {
        panic!("CMake configure failed");
    }

    // Apply Metal version patch after CMake fetches the sources
    patch_metal_version(&out_dir);

    let status = Command::new("cmake")
        .args([
            "--build",
            &build_dir.to_string_lossy(),
            "--config",
            "Release",
            "-j",
        ])
        .status()
        .expect("Failed to run cmake build");

    if !status.success() {
        panic!("CMake build failed");
    }

    // Link the libraries
    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!(
        "cargo:rustc-link-search=native={}/_deps/mlx-build",
        build_dir.display()
    );
    println!("cargo:rustc-link-lib=static=mlx");
    println!("cargo:rustc-link-lib=static=mlxc");

    // Link gguflib if present
    let gguflib_dir = build_dir.join("_deps/mlx-build/mlx/io");
    if gguflib_dir.join("libgguflib.a").exists() {
        println!(
            "cargo:rustc-link-search=native={}",
            gguflib_dir.display()
        );
        println!("cargo:rustc-link-lib=static=gguflib");
    }

    // Copy metallib to target directory for runtime discovery
    let metallib = build_dir.join("_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib");
    copy_metallib_to_target_dir(&metallib);
}

fn link_system_frameworks() {
    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=dylib=objc");
    println!("cargo:rustc-link-lib=framework=Foundation");

    #[cfg(feature = "metal")]
    {
        println!("cargo:rustc-link-lib=framework=Metal");
    }

    #[cfg(feature = "accelerate")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}

// ─── Main ────────────────────────────────────────────────────────

fn main() {
    println!("cargo:rerun-if-env-changed=MLX_PREBUILT_PATH");

    std::env::set_var("MACOSX_DEPLOYMENT_TARGET", "15.0");
    println!("cargo:rustc-link-arg=-mmacosx-version-min=15.0");

    // Three-mode detection:
    // 1. Explicit MLX_PREBUILT_PATH env var
    // 2. Metal compiler available → build from source
    // 3. Metal compiler unavailable → auto-download prebuilt
    if let Some(path) = explicit_prebuilt_path() {
        println!(
            "cargo:warning=Using explicit pre-built MLX from: {}",
            path.display()
        );
        link_prebuilt(&path);
    } else if metal_compiler_available() {
        println!("cargo:warning=Metal compiler found, building MLX from source");
        build_from_source();
    } else {
        println!("cargo:warning=Metal compiler not found, downloading pre-built MLX");
        let prebuilt_dir = download_prebuilt();
        link_prebuilt(&prebuilt_dir);
    }

    link_system_frameworks();

    // Generate bindings (works in all modes — headers come from git submodule)
    let bindings = bindgen::Builder::default()
        .rust_target("1.73.0".parse().expect("rust-version"))
        .header("src/mlx-c/mlx/c/mlx.h")
        .header("src/mlx-c/mlx/c/linalg.h")
        .header("src/mlx-c/mlx/c/error.h")
        .header("src/mlx-c/mlx/c/transforms_impl.h")
        .clang_arg("-Isrc/mlx-c")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
