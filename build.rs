//! Release identity plumbing: compile-time git provenance for the UCI
//! handshake and artifact manifests.
//!
//! Provenance resolution order:
//!   1. Explicit `EUREKA_GIT_*` build metadata env vars (frozen by the release
//!      builder from the SOURCE checkout and injected wherever the binary is
//!      actually compiled — `git archive`, container, or cross build — none of
//!      which carry a `.git`). This keeps release provenance trustworthy even
//!      when git is absent.
//!   2. Fall back to the local git checkout (dev builds).
//!
//! If neither source is available the fields degrade to "unknown"/"false",
//! which only ever happens for a build with no provenance at all.

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

/// `EUREKA_GIT_*` env override, else `git`, else the fallback default.
fn resolve(env_key: &str, git_value: String, fallback: &str) -> String {
    if let Ok(v) = std::env::var(env_key) {
        if !v.is_empty() {
            return v;
        }
    }
    if !git_value.is_empty() {
        return git_value;
    }
    fallback.to_string()
}

fn resolve_bool(env_key: &str, git_value: bool) -> bool {
    if let Ok(v) = std::env::var(env_key) {
        return v == "true" || v == "1";
    }
    git_value
}

fn main() {
    // Explicit provenance injected by the release builder.
    for key in [
        "EUREKA_GIT_SHA",
        "EUREKA_GIT_SHA_SHORT",
        "EUREKA_GIT_TAG",
        "EUREKA_GIT_DATE",
        "EUREKA_GIT_DIRTY",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
    }

    // Re-run build.rs (and therefore re-derive the dirty flag) when anything
    // that actually feeds the binary changes. `.git/index` alone only moves on
    // stage/commit; unstaged edits to `src/` would otherwise keep a stale
    // dirty=false identity cached in the binary. Deliberately NOT
    // `rerun-if-changed=.` (that would loop on `target/`).
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=Cargo.lock");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs/");
    println!("cargo:rerun-if-changed=.git/index");

    // Resolve each field: explicit env override -> git -> degraded default.
    let full = resolve(
        "EUREKA_GIT_SHA",
        git(&["rev-parse", "HEAD"]).unwrap_or_default(),
        "unknown",
    );
    let short = resolve(
        "EUREKA_GIT_SHA_SHORT",
        full.get(..8).unwrap_or("unknown").to_string(),
        "unknown",
    );
    let dirty = resolve_bool(
        "EUREKA_GIT_DIRTY",
        Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .map(|o| !o.stdout.is_empty())
            .unwrap_or(false),
    );
    let tag = resolve(
        "EUREKA_GIT_TAG",
        git(&["describe", "--tags", "--exact-match", "HEAD"]).unwrap_or_default(),
        "",
    );
    let date = resolve(
        "EUREKA_GIT_DATE",
        git(&["show", "-s", "--format=%cs", "HEAD"]).unwrap_or_default(),
        "unknown",
    );

    println!("cargo:rustc-env=GIT_SHA={full}");
    println!("cargo:rustc-env=GIT_SHA_SHORT={short}");
    println!("cargo:rustc-env=GIT_DIRTY={dirty}");
    println!("cargo:rustc-env=GIT_TAG={tag}");
    println!("cargo:rustc-env=GIT_DATE={date}");
}
