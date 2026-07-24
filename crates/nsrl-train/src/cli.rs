//! Lightweight CLI helpers shared across the ~30 `nsrl-train` binaries.
//!
//! Every binary previously duplicated a manual argument-parsing loop, the
//! `required()` helper, and the `run()` / `main` error-reporting pattern.
//! This module factors those out so the binaries can stay focused on their
//! actual work.
//!
//! # Usage
//!
//! ```rust,ignore
//! use nsrl_train::cli::{next_arg, run_main};
//!
//! fn main() {
//!     run_main(run)
//! }
//!
//! fn run() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut args = std::env::args().skip(1);
//!     while let Some(arg) = args.next() {
//!         match arg.as_str() {
//!             "--model" => model_path = Some(next_arg(&mut args, "--model")?.into()),
//!             // ...
//!             _ => return Err(format!("unknown flag: {arg}").into()),
//!         }
//!     }
//!     // ...
//!     Ok(())
//! }
//! ```

use std::path::PathBuf;

/// Return the next argument from `args`, or an error if there isn't one.
///
/// Callers typically convert the returned `String` into whatever type they
/// need (e.g. `.parse()?` for numbers, `.into()` for `PathBuf`).
pub fn next_arg(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value").into())
}

/// Return the contained value, or an error if it's `None`.
///
/// This is the "already-parsed Option" variant used by binaries that collect
/// options into `Option<PathBuf>` and then validate them all at once.
pub fn required<T>(value: Option<T>, flag: &str) -> Result<T, Box<dyn std::error::Error>> {
    value.ok_or_else(|| format!("{flag} is required").into())
}

/// Standard entry-point wrapper: calls `f()`, prints any error to stderr, and
/// exits with code 2 on failure (matching the existing convention across all
/// binaries).
pub fn run_main(f: fn() -> Result<(), Box<dyn std::error::Error>>) {
    if let Err(error) = f() {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}

/// Parse a required `--flag <value>` argument, validating that the value exists.
///
/// Convenience wrapper around `next_arg` + `.into()` for the most common case
/// where the flag value becomes a `PathBuf`.
pub fn required_path(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(PathBuf::from(next_arg(args, flag)?))
}
