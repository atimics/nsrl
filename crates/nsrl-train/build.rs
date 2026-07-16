use std::fs;
use std::path::{Path, PathBuf};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn main() {
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let inputs = [
        manifest.join("Cargo.toml"),
        manifest.join("build.rs"),
        manifest.join("src"),
        manifest.join("../nsrl-core/Cargo.toml"),
        manifest.join("../nsrl-core/src"),
        manifest.join("../nsrl-train-core/Cargo.toml"),
        manifest.join("../nsrl-train-core/src"),
    ];
    let mut files = Vec::new();
    for input in &inputs {
        println!("cargo:rerun-if-changed={}", input.display());
        collect_files(input, &mut files);
    }
    files.sort();
    let mut hash = FNV_OFFSET;
    for path in files {
        let relative = path.strip_prefix(&manifest).unwrap_or(&path);
        hash = hash_bytes(hash, relative.to_string_lossy().as_bytes());
        hash = hash_bytes(hash, &[0]);
        hash = hash_bytes(hash, &fs::read(&path).expect("read source binding input"));
        hash = hash_bytes(hash, &[0xff]);
    }
    println!("cargo:rustc-env=NSRL_BOOLEAN_JET_SOURCE_FNV64={hash}");
}

fn collect_files(path: &Path, files: &mut Vec<PathBuf>) {
    if path.is_file() {
        files.push(path.to_path_buf());
        return;
    }
    let mut entries = fs::read_dir(path)
        .expect("read source binding directory")
        .map(|entry| entry.expect("read source binding entry").path())
        .collect::<Vec<_>>();
    entries.sort();
    for entry in entries {
        if entry.is_dir() {
            collect_files(&entry, files);
        } else if entry.extension().is_some_and(|extension| extension == "rs") {
            files.push(entry);
        }
    }
}

fn hash_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash = (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
    }
    hash
}
