use std::path::{Path, PathBuf};

/// Resolve a model specifier to a concrete directory path.
///
/// 1. If `path` is an existing directory, returns it directly.
/// 2. If it looks like `org/name`, resolves from the HuggingFace cache
///    (`~/.cache/huggingface/hub/models--org--name/snapshots/<hash>`).
pub fn resolve(path: &str) -> Result<PathBuf, String> {
    let as_path = Path::new(path);

    let expanded = if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = directories::BaseDirs::new().map(|d| d.home_dir().to_path_buf()) {
            home.join(rest)
        } else {
            as_path.to_path_buf()
        }
    } else {
        as_path.to_path_buf()
    };

    if expanded.is_dir() {
        return Ok(expanded);
    }

    resolve_with_cache(path, default_hf_cache().as_deref())
}

/// Testable resolver with explicit cache root.
fn resolve_with_cache(path: &str, cache_root: Option<&Path>) -> Result<PathBuf, String> {
    let as_path = Path::new(path);
    if as_path.is_dir() {
        return Ok(as_path.to_path_buf());
    }

    if let Some((org, name)) = path.split_once('/') {
        if !org.is_empty() && !name.is_empty() && !name.contains('/') {
            if let Some(cache) = cache_root {
                return resolve_hf_snapshot(cache, org, name);
            }
        }
    }

    Err(format!(
        "model '{path}' is not an existing directory and was not found in the HuggingFace cache"
    ))
}

/// Read `refs/main` and resolve to the snapshot directory.
fn resolve_hf_snapshot(cache_root: &Path, org: &str, name: &str) -> Result<PathBuf, String> {
    let model_dir = cache_root.join(format!("models--{org}--{name}"));
    let ref_path = model_dir.join("refs").join("main");
    let hash = std::fs::read_to_string(&ref_path)
        .map_err(|e| format!("could not read HF cache ref for '{org}/{name}': {e}"))?
        .trim()
        .to_owned();

    if hash.is_empty() {
        return Err(format!("empty ref in HF cache for '{org}/{name}'"));
    }

    let snapshot_dir = model_dir.join("snapshots").join(&hash);
    if snapshot_dir.is_dir() {
        Ok(snapshot_dir)
    } else {
        Err(format!(
            "snapshot directory missing for '{org}/{name}' (hash: {hash})"
        ))
    }
}

fn default_hf_cache() -> Option<PathBuf> {
    // HuggingFace Python uses $HF_HOME/hub or ~/.cache/huggingface/hub,
    // NOT the platform-specific cache dir (~/Library/Caches on macOS).
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return Some(PathBuf::from(hf_home).join("hub"));
    }
    directories::BaseDirs::new()
        .map(|d| d.home_dir().join(".cache").join("huggingface").join("hub"))
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    fn create_hf_cache(root: &Path, org: &str, name: &str, hash: &str) -> PathBuf {
        let model_dir = root.join(format!("models--{org}--{name}"));
        let refs_dir = model_dir.join("refs");
        let snapshot_dir = model_dir.join("snapshots").join(hash);
        std::fs::create_dir_all(&refs_dir).unwrap();
        std::fs::create_dir_all(&snapshot_dir).unwrap();
        std::fs::write(refs_dir.join("main"), hash).unwrap();
        snapshot_dir
    }

    #[test]
    fn test_resolve_existing_directory_returns_as_is() {
        let dir = tempfile::tempdir().unwrap();
        let result = resolve_with_cache(dir.path().to_str().unwrap(), None);
        assert_eq!(result.unwrap(), dir.path());
    }

    #[test]
    fn test_resolve_hf_model_id_from_cache() {
        let cache = tempfile::tempdir().unwrap();
        let snapshot = create_hf_cache(cache.path(), "mlx-community", "Qwen3-4bit", "abc123");
        let result = resolve_with_cache("mlx-community/Qwen3-4bit", Some(cache.path()));
        assert_eq!(result.unwrap(), snapshot);
    }

    #[test]
    fn test_resolve_hf_model_id_not_in_cache_is_err() {
        let cache = tempfile::tempdir().unwrap();
        let result = resolve_with_cache("no-org/NoModel", Some(cache.path()));
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_hf_empty_refs_main_is_err() {
        let cache = tempfile::tempdir().unwrap();
        let model_dir = cache.path().join("models--org--name");
        let refs_dir = model_dir.join("refs");
        std::fs::create_dir_all(&refs_dir).unwrap();
        std::fs::write(refs_dir.join("main"), "  \n").unwrap();
        let result = resolve_with_cache("org/name", Some(cache.path()));
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_hf_snapshot_dir_missing_is_err() {
        let cache = tempfile::tempdir().unwrap();
        let model_dir = cache.path().join("models--org--name");
        let refs_dir = model_dir.join("refs");
        std::fs::create_dir_all(&refs_dir).unwrap();
        std::fs::write(refs_dir.join("main"), "deadbeef").unwrap();
        let result = resolve_with_cache("org/name", Some(cache.path()));
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_nonexistent_plain_path_is_err() {
        let result = resolve_with_cache("/nonexistent/path/to/model", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_no_cache_for_hf_id_is_err() {
        let result = resolve_with_cache("org/model", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_hf_hash_with_trailing_newline() {
        let cache = tempfile::tempdir().unwrap();
        let snapshot = create_hf_cache(cache.path(), "org", "model", "abc123");
        // Overwrite with trailing newline (common from `echo`)
        let ref_path = cache
            .path()
            .join("models--org--model")
            .join("refs")
            .join("main");
        std::fs::write(&ref_path, "abc123\n").unwrap();
        let result = resolve_with_cache("org/model", Some(cache.path()));
        assert_eq!(result.unwrap(), snapshot);
    }
}
