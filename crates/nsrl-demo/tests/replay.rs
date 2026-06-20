use std::fs;
use std::process::Command;

#[test]
fn demo_binary_replays_byte_identical_trace() {
    let bin = env!("CARGO_BIN_EXE_nsrl-demo");
    let base = std::env::temp_dir().join(format!(
        "nsrl-demo-replay-{}-{}",
        std::process::id(),
        "abba"
    ));
    let left = base.with_extension("left.jsonl");
    let right = base.with_extension("right.jsonl");

    let left_status = Command::new(bin)
        .args(["--input", "abba", "--trace"])
        .arg(&left)
        .status()
        .expect("run left demo");
    assert!(left_status.success());

    let right_status = Command::new(bin)
        .args(["--input", "abba", "--trace"])
        .arg(&right)
        .status()
        .expect("run right demo");
    assert!(right_status.success());

    let left_bytes = fs::read(&left).expect("read left trace");
    let right_bytes = fs::read(&right).expect("read right trace");

    let _ = fs::remove_file(&left);
    let _ = fs::remove_file(&right);

    assert_eq!(left_bytes, right_bytes);

    let text = String::from_utf8(left_bytes).expect("trace utf8");
    assert!(text.contains("\"schema\":\"nsrl.forward_trace.v1\""));
    assert!(text.contains("\"authority\":\"integer_runtime_determinism\""));
    assert!(text.contains("\"output_hash\""));
}
