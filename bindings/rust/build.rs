// Resolves the libvietasr native library at build time.
//
// Order: VIETASR_NATIVE_DIR override -> bundled `_native/` (source checkout) ->
// download the matching bundle (~67 MB, ONNX model embedded) from the crate's
// GitHub Release. The crate itself ships source only — crates.io caps packages
// at ~10 MB.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const REPO: &str = "dangvansam/viet-asr";

fn platform_key() -> &'static str {
    let target = env::var("TARGET").unwrap_or_default();
    if target.contains("apple-darwin") {
        "darwin-universal2"
    } else if target.contains("windows") {
        "win-x64"
    } else if target.contains("linux") {
        if target.starts_with("aarch64") || target.starts_with("arm") {
            "linux-arm64"
        } else {
            "linux-x64"
        }
    } else {
        panic!("viet-asr: unsupported target '{target}'");
    }
}

fn lib_present(dir: &Path) -> bool {
    ["libvietasr.so", "libvietasr.dylib", "vietasr.dll"]
        .iter()
        .any(|name| dir.join(name).exists())
}

fn download_native(out_dir: &Path) -> PathBuf {
    let dir = out_dir.join("native");
    fs::create_dir_all(&dir).expect("create native dir");
    if lib_present(&dir) {
        return dir;
    }

    let asset = format!("viet-asr-native-{}.tar.gz", platform_key());
    let version = env::var("CARGO_PKG_VERSION").unwrap();
    let tarball = dir.join(&asset);
    let urls = [
        format!("https://github.com/{REPO}/releases/download/v{version}/{asset}"),
        format!("https://github.com/{REPO}/releases/latest/download/{asset}"),
    ];

    for url in &urls {
        let fetched = Command::new("curl")
            .args(["-fSL", "--retry", "3", "-o"])
            .arg(&tarball)
            .arg(url)
            .status();
        if !matches!(fetched, Ok(s) if s.success()) {
            continue;
        }
        let extracted = Command::new("tar")
            .arg("-xzf")
            .arg(&tarball)
            .arg("-C")
            .arg(&dir)
            .status();
        if matches!(extracted, Ok(s) if s.success()) && lib_present(&dir) {
            let _ = fs::remove_file(&tarball);
            return dir;
        }
    }
    panic!(
        "viet-asr: could not download the native library for {}. \
         Set VIETASR_NATIVE_DIR to a directory containing libvietasr.",
        platform_key()
    );
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let native_dir = if let Ok(dir) = env::var("VIETASR_NATIVE_DIR") {
        PathBuf::from(dir)
    } else if lib_present(&manifest_dir.join("_native")) {
        manifest_dir.join("_native")
    } else {
        download_native(&out_dir)
    };

    println!("cargo:rustc-link-search=native={}", native_dir.display());
    println!("cargo:rustc-link-lib=dylib=vietasr");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", native_dir.display());
    }
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=VIETASR_NATIVE_DIR");
}
