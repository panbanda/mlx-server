//! CLI integration tests for `higgs exec`.

#![allow(clippy::panic, clippy::unwrap_used, clippy::tests_outside_test_module)]

use std::io::Write;
use std::net::TcpListener;
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
    // Bind a random port then drop the listener so nothing is listening on it.
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();

    let dir = write_test_config(port);

    let output = Command::new(higgs_bin())
        .args(["exec", "--", "echo", "hello"])
        .env("HIGGS_CONFIG_DIR", dir.path())
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

/// Write a minimal valid config.toml pointing at the given port.
/// Includes a dummy provider so config validation passes (otherwise
/// `load_config_for_command` errors and `unwrap_or_default` silently
/// falls back to defaults, making the test flaky).
fn write_test_config(port: u16) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("config.toml");
    let mut f = std::fs::File::create(&config_path).unwrap();
    write!(
        f,
        "[server]\nhost = \"127.0.0.1\"\nport = {port}\n\n\
         [provider.dummy]\nurl = \"http://127.0.0.1:1\"\n"
    )
    .unwrap();
    dir
}

/// Bind a random port and write a config pointing at it.
/// Returns (`TcpListener`, temp dir) -- keep the listener alive so `higgs exec`
/// sees a "running" server.
fn fake_server_env() -> (TcpListener, tempfile::TempDir) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let dir = write_test_config(port);
    (listener, dir)
}

#[test]
fn exec_forwards_child_exit_code() {
    let (_listener, dir) = fake_server_env();

    let output = Command::new(higgs_bin())
        .args(["exec", "--", "sh", "-c", "exit 42"])
        .env("HIGGS_CONFIG_DIR", dir.path())
        .output()
        .unwrap();

    assert_eq!(
        output.status.code(),
        Some(42),
        "expected exit code 42, got {:?}",
        output.status.code()
    );
}

#[test]
fn exec_forwards_zero_exit_code() {
    let (_listener, dir) = fake_server_env();

    let output = Command::new(higgs_bin())
        .args(["exec", "--", "true"])
        .env("HIGGS_CONFIG_DIR", dir.path())
        .output()
        .unwrap();

    assert_eq!(
        output.status.code(),
        Some(0),
        "expected exit code 0, got {:?}",
        output.status.code()
    );
}

#[test]
fn exec_reports_spawn_failure() {
    let (_listener, dir) = fake_server_env();

    let output = Command::new(higgs_bin())
        .args(["exec", "--", "/nonexistent/binary"])
        .env("HIGGS_CONFIG_DIR", dir.path())
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("failed to spawn"),
        "expected 'failed to spawn' in stderr, got: {stderr}"
    );
}

#[test]
fn exec_sigint_terminates_child() {
    use std::time::{Duration, Instant};

    let (_listener, dir) = fake_server_env();

    let mut child = Command::new(higgs_bin())
        .args(["exec", "--", "sleep", "60"])
        .env("HIGGS_CONFIG_DIR", dir.path())
        .spawn()
        .unwrap();

    // Give the child process time to start
    std::thread::sleep(Duration::from_millis(500));

    let pid = nix::unistd::Pid::from_raw(i32::try_from(child.id()).unwrap());
    nix::sys::signal::kill(pid, nix::sys::signal::Signal::SIGINT).unwrap();

    let start = Instant::now();
    let status = child.wait().unwrap();
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_secs(5),
        "child should terminate within 5s after SIGINT, took {elapsed:?}"
    );
    // SIGINT handler forwards SIGTERM (signal 15) to child, so exit code
    // should be 128 + 15 = 143 per Unix convention.
    assert_eq!(
        status.code(),
        Some(143),
        "expected exit code 143 (128 + SIGTERM), got: {status}"
    );
}
