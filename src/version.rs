//! Release identity (R0 - Eureka v0.1.0).
//!
//! Every binary reports, through the UCI handshake, its exact version and
//! provenance so it can never again be an anonymous "target/release exe".
//!
//! Version rules (frozen):
//!   tagged release        -> `0.1.0`          (id: Eureka v0.1.0)
//!   ordinary dev HEAD     -> `0.1.0-dev+<sha>`
//!   dirty worktree        -> `0.1.0-dev+<sha>.dirty`
//!
//! `CurrentFinal` is a search-policy name, NEVER a product version name.

/// Cargo package version (semver; the product version is derived from it).
pub const PKG_VERSION: &str = env!("CARGO_PKG_VERSION");

fn git_short() -> &'static str {
    option_env!("GIT_SHA_SHORT").unwrap_or("unknown")
}

fn git_full() -> &'static str {
    option_env!("GIT_SHA").unwrap_or("unknown")
}

fn git_date() -> &'static str {
    option_env!("GIT_DATE").unwrap_or("unknown")
}

fn exact_tag() -> &'static str {
    option_env!("GIT_TAG").unwrap_or("")
}

/// True when the build worktree had uncommitted changes.
pub fn is_dirty() -> bool {
    option_env!("GIT_DIRTY").unwrap_or("false") == "true"
}

/// `0.1.0` for a tagged release, `0.1.0-dev+<sha>[.dirty]` otherwise.
pub fn version_string() -> String {
    if !exact_tag().is_empty() {
        return PKG_VERSION.to_string();
    }
    let mut v = format!("{PKG_VERSION}-dev+{}", git_short());
    if is_dirty() {
        v.push_str(".dirty");
    }
    v
}

/// `eureka-0.1.0-2026-08-13-<sha>-<platform>`.
pub fn build_string() -> String {
    format!(
        "eureka-{}-{}-{}-{}",
        version_string(),
        git_date(),
        git_short(),
        platform_string(),
    )
}

pub fn source_sha() -> &'static str {
    git_full()
}

pub fn release_date() -> &'static str {
    git_date()
}

pub fn platform_string() -> String {
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_string_shape() {
        let v = version_string();
        assert!(v.starts_with(PKG_VERSION), "starts with semver: {v}");
        if exact_tag().is_empty() {
            assert!(v.contains("-dev+"), "dev build carries -dev+<sha>: {v}");
        }
    }

    #[test]
    fn build_string_shape() {
        let b = build_string();
        assert!(
            b.starts_with("eureka-"),
            "build id starts with eureka-: {b}"
        );
        assert!(b.ends_with(&platform_string()));
    }
}
