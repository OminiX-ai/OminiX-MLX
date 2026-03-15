fn main() {
    // Link CoreGraphics framework on macOS for PDF rendering
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=CoreGraphics");
    }
}
