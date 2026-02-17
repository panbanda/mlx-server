use std::collections::HashMap;
use std::time::Instant;

use mlx_models::cache::ConcatKeyValueCache;

/// In-memory prefix KV cache with LRU eviction.
///
/// Caches KV states for token prefixes (typically system prompts) so that
/// subsequent requests sharing the same prefix can skip the prefill phase
/// for the cached tokens.
pub struct PrefixCache {
    entries: HashMap<u64, CacheEntry>,
    max_entries: usize,
}

struct CacheEntry {
    prefix_tokens: Vec<u32>,
    kv_cache: Vec<Option<ConcatKeyValueCache>>,
    last_accessed: Instant,
}

/// Result of a prefix cache lookup.
pub struct PrefixMatch {
    /// Number of tokens from the beginning that matched the cached prefix.
    pub prefix_len: usize,
    /// Cloned KV cache state for the matched prefix.
    pub kv_cache: Vec<Option<ConcatKeyValueCache>>,
}

impl PrefixCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries,
        }
    }

    /// Find the longest cached prefix that matches the beginning of `tokens`.
    ///
    /// Returns `None` if no prefix matches or if the match is too short to
    /// be worth reusing (less than 16 tokens).
    pub fn find_longest_prefix(&mut self, tokens: &[u32]) -> Option<PrefixMatch> {
        let min_prefix_len = 16;
        let mut best_key: Option<u64> = None;
        let mut best_len: usize = 0;

        for (key, entry) in &self.entries {
            let prefix = &entry.prefix_tokens;
            if prefix.len() > tokens.len() {
                continue;
            }
            if prefix.len() < min_prefix_len {
                continue;
            }

            // Check if the cached prefix matches the beginning of the new tokens
            let matches = prefix.iter().zip(tokens.iter()).all(|(a, b)| a == b);

            if matches && prefix.len() > best_len {
                best_len = prefix.len();
                best_key = Some(*key);
            }
        }

        if let Some(key) = best_key
            && let Some(entry) = self.entries.get_mut(&key)
        {
            entry.last_accessed = Instant::now();
            return Some(PrefixMatch {
                prefix_len: entry.prefix_tokens.len(),
                kv_cache: entry.kv_cache.clone(),
            });
        }

        None
    }

    /// Store a prefix and its KV cache state.
    pub fn store(&mut self, prefix_tokens: Vec<u32>, kv_cache: Vec<Option<ConcatKeyValueCache>>) {
        if self.max_entries == 0 {
            return;
        }

        // Evict if at capacity
        if self.entries.len() >= self.max_entries {
            self.evict_lru();
        }

        let key = hash_tokens(&prefix_tokens);
        self.entries.insert(
            key,
            CacheEntry {
                prefix_tokens,
                kv_cache,
                last_accessed: Instant::now(),
            },
        );
    }

    /// Remove the least recently used entry.
    fn evict_lru(&mut self) {
        let oldest_key = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(key, _)| *key);

        if let Some(key) = oldest_key {
            self.entries.remove(&key);
        }
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all cached entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// Hash a token sequence for use as a cache key.
fn hash_tokens(tokens: &[u32]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    tokens.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use mlx_models::cache::ConcatKeyValueCache;

    fn make_dummy_kv_cache(num_layers: usize) -> Vec<Option<ConcatKeyValueCache>> {
        (0..num_layers)
            .map(|_| Some(ConcatKeyValueCache::new()))
            .collect()
    }

    #[test]
    fn test_empty_cache_returns_none() {
        let mut cache = PrefixCache::new(10);
        assert!(cache.find_longest_prefix(&[1, 2, 3]).is_none());
        assert!(cache.is_empty());
    }

    #[test]
    fn test_store_and_find_exact_match() {
        let mut cache = PrefixCache::new(10);
        let prefix: Vec<u32> = (0..32).collect();
        let kv = make_dummy_kv_cache(4);

        cache.store(prefix.clone(), kv);
        assert_eq!(cache.len(), 1);

        // Search with the same prefix followed by more tokens
        let mut query: Vec<u32> = prefix;
        query.extend_from_slice(&[100, 101, 102]);

        let result = cache.find_longest_prefix(&query);
        assert!(result.is_some());
        let matched = result.unwrap();
        assert_eq!(matched.prefix_len, 32);
        assert_eq!(matched.kv_cache.len(), 4);
    }

    #[test]
    fn test_no_match_for_different_prefix() {
        let mut cache = PrefixCache::new(10);
        let prefix: Vec<u32> = (0..32).collect();
        cache.store(prefix, make_dummy_kv_cache(4));

        // Different token sequence
        let query: Vec<u32> = (100..132).collect();
        assert!(cache.find_longest_prefix(&query).is_none());
    }

    #[test]
    fn test_prefix_too_short_ignored() {
        let mut cache = PrefixCache::new(10);
        // Only 8 tokens (below min threshold of 16)
        let prefix: Vec<u32> = (0..8).collect();
        cache.store(prefix, make_dummy_kv_cache(4));

        let query: Vec<u32> = (0..32).collect();
        assert!(cache.find_longest_prefix(&query).is_none());
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = PrefixCache::new(2);

        let prefix_a: Vec<u32> = (0..32).collect();
        let prefix_b: Vec<u32> = (100..132).collect();
        let prefix_c: Vec<u32> = (200..232).collect();

        cache.store(prefix_a, make_dummy_kv_cache(4));
        cache.store(prefix_b, make_dummy_kv_cache(4));
        assert_eq!(cache.len(), 2);

        // Adding a third should evict the LRU (prefix_a, since it was stored first)
        cache.store(prefix_c.clone(), make_dummy_kv_cache(4));
        assert_eq!(cache.len(), 2);

        // prefix_c should still be found
        let mut query_c: Vec<u32> = prefix_c;
        query_c.push(999);
        assert!(cache.find_longest_prefix(&query_c).is_some());
    }

    #[test]
    fn test_zero_capacity_never_stores() {
        let mut cache = PrefixCache::new(0);
        let prefix: Vec<u32> = (0..32).collect();
        cache.store(prefix.clone(), make_dummy_kv_cache(4));
        assert!(cache.is_empty());

        let mut query = prefix;
        query.push(999);
        assert!(cache.find_longest_prefix(&query).is_none());
    }

    #[test]
    fn test_clear() {
        let mut cache = PrefixCache::new(10);
        let prefix: Vec<u32> = (0..32).collect();
        cache.store(prefix, make_dummy_kv_cache(4));
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_longest_prefix_wins() {
        let mut cache = PrefixCache::new(10);

        // Store a shorter prefix
        let short_prefix: Vec<u32> = (0..20).collect();
        cache.store(short_prefix, make_dummy_kv_cache(4));

        // Store a longer prefix that extends the short one
        let long_prefix: Vec<u32> = (0..50).collect();
        cache.store(long_prefix, make_dummy_kv_cache(4));

        // Query should match the longer prefix
        let query: Vec<u32> = (0..64).collect();
        let result = cache.find_longest_prefix(&query);
        assert!(result.is_some());
        assert_eq!(result.unwrap().prefix_len, 50);
    }
}
