use std::env;
use std::fs;
use std::path::PathBuf;

const GENERATED_LUT_SOURCE: &str = include_str!("src/rsqrt_lut_8bit.rs");

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/rsqrt_lut_8bit.rs");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    fs::write(out_dir.join("rsqrt_lut_8bit.rs"), GENERATED_LUT_SOURCE)
        .expect("write generated LUT tables");
}
