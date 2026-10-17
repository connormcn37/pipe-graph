//! Exercises the `pipe-graph` binary itself (not just the library) through
//! its `run`/`check` subcommands.

use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pipe-graph"))
}

#[test]
fn run_executes_a_graph_file() {
    let out = bin()
        .args([
            "run",
            "examples/graphs/split_merge.graph",
            "--input",
            "src.in=4x4x3:200",
        ])
        .output()
        .expect("spawn pipe-graph");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("4 nodes, 5 edges, 1 tick(s)"),
        "stdout: {stdout}"
    );
    assert!(
        stdout.contains("crop.out: Frame 2x2x3 U8"),
        "stdout: {stdout}"
    );
}

#[test]
fn check_validates_without_running() {
    let out = bin()
        .args(["check", "examples/graphs/split_merge.graph"])
        .output()
        .expect("spawn pipe-graph");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("ok: examples/graphs/split_merge.graph: 4 nodes, 5 edges"));
}

#[test]
fn run_reports_a_missing_file() {
    let out = bin()
        .args(["run", "no-such-file.graph"])
        .output()
        .expect("spawn pipe-graph");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("no-such-file.graph"), "stderr: {stderr}");
}

#[test]
fn unknown_subcommand_is_an_error() {
    let out = bin().args(["bogus"]).output().expect("spawn pipe-graph");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("unknown subcommand"), "stderr: {stderr}");
}

#[test]
fn no_args_runs_the_demo() {
    let out = bin().output().expect("spawn pipe-graph");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("round-trip ok: true"), "stdout: {stdout}");
}
