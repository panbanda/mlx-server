//! CLI integration tests for `higgs exec`.

#![allow(clippy::panic, clippy::unwrap_used, clippy::tests_outside_test_module)]

use std::process::Command;

fn higgs_bin() -> std::path::PathBuf {
    // cargo sets this during `cargo test`
    let mut path = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    path.push("higgs");
    path
}

#[test]
fn exec_exits_with_error_when_server_not_running() {
    let output = Command::new(higgs_bin())
        .args(["exec", "--", "echo", "hello"])
        .env("HIGGS_CONFIG_DIR", "/tmp/higgs-test-nonexistent")
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("not running"),
        "expected 'not running' in stderr, got: {stderr}"
    );
}

#[test]
fn exec_requires_command_argument() {
    let output = Command::new(higgs_bin()).args(["exec"]).output().unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    // clap should report missing required argument
    assert!(
        stderr.contains("required") || stderr.contains("Usage"),
        "expected clap error in stderr, got: {stderr}"
    );
}
