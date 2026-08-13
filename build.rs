//! Release identity plumbing: compile-time git provenance for the UCI
//! handshake and artifact manifests. Runs git on the worktree so any exe
//! reports its exact source SHA, dirty state, tag and commit date.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs/");

    let full = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    let short = full.get(..8).unwrap_or("unknown").to_string();
    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);
    let tag = git(&["describe", "--tags", "--exact-match", "HEAD"]).unwrap_or_default();
    let date =
        git(&["show", "-s", "--format=%cs", "HEAD"]).unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=GIT_SHA={full}");
    println!("cargo:rustc-env=GIT_SHA_SHORT={short}");
    println!("cargo:rustc-env=GIT_DIRTY={dirty}");
    println!("cargo:rustc-env=GIT_TAG={tag}");
    println!("cargo:rustc-env=GIT_DATE={date}");
}
