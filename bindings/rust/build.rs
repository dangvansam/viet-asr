use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    let native_dir = env::var("VIETASR_NATIVE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest_dir.join("_native"));

    println!("cargo:rustc-link-search=native={}", native_dir.display());
    println!("cargo:rustc-link-lib=dylib=vietasr");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", native_dir.display());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=VIETASR_NATIVE_DIR");
}
