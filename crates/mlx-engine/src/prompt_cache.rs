use std::collections::HashMap;
use std::time::Instant;

use mlx_models::AnyCache;

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
    cache: AnyCache,
    last_accessed: Instant,
}

/// Result of a prefix cache lookup.
pub struct PrefixMatch {
    /// Number of tokens from the beginning that matched the cached prefix.
    pub prefix_len: usize,
    /// Cloned cache state for the matched prefix.
    pub cache: AnyCache,
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

        if let Some(key) = best_key {
            if let Some(entry) = self.entries.get_mut(&key) {
                entry.last_accessed = Instant::now();
                return Some(PrefixMatch {
                    prefix_len: entry.prefix_tokens.len(),
                    cache: entry.cache.clone(),
                });
            }
        }

        None
    }

    /// Store a prefix and its cache state.
    pub fn store(&mut self, prefix_tokens: Vec<u32>, cache: AnyCache) {
        if self.max_entries == 0 {
            return;
        }

        let key = hash_tokens(&prefix_tokens);

        // Skip if key collides with a different token sequence
        if let Some(existing) = self.entries.get(&key) {
            if existing.prefix_tokens != prefix_tokens {
                return;
            }
        }

        if !self.entries.contains_key(&key) && self.entries.len() >= self.max_entries {
            self.evict_lru();
        }

        self.entries.insert(
            key,
            CacheEntry {
                prefix_tokens,
                cache,
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
    use mlx_models::AnyCache;
    use mlx_models::cache::ConcatKeyValueCache;

    fn make_dummy_cache(num_layers: usize) -> AnyCache {
        let kv: Vec<Option<ConcatKeyValueCache>> = (0..num_layers)
            .map(|_| Some(ConcatKeyValueCache::new()))
            .collect();
        AnyCache::KV(kv)
    }

    fn cache_layer_count(cache: &AnyCache) -> usize {
        match cache {
            AnyCache::KV(v) => v.len(),
            AnyCache::Hybrid(v) => v.len(),
        }
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
        let kv = make_dummy_cache(4);

        cache.store(prefix.clone(), kv);
        assert_eq!(cache.len(), 1);

        // Search with the same prefix followed by more tokens
        let mut query: Vec<u32> = prefix;
        query.extend_from_slice(&[100, 101, 102]);

        let result = cache.find_longest_prefix(&query);
        assert!(result.is_some());
        let matched = result.unwrap();
        assert_eq!(matched.prefix_len, 32);
        assert_eq!(cache_layer_count(&matched.cache), 4);
    }

    #[test]
    fn test_no_match_for_different_prefix() {
        let mut cache = PrefixCache::new(10);
        let prefix: Vec<u32> = (0..32).collect();
        cache.store(prefix, make_dummy_cache(4));

        // Different token sequence
        let query: Vec<u32> = (100..132).collect();
        assert!(cache.find_longest_prefix(&query).is_none());
    }

    #[test]
    fn test_prefix_too_short_ignored() {
        let mut cache = PrefixCache::new(10);
        // Only 8 tokens (below min threshold of 16)
        let prefix: Vec<u32> = (0..8).collect();
        cache.store(prefix, make_dummy_cache(4));

        let query: Vec<u32> = (0..32).collect();
        assert!(cache.find_longest_prefix(&query).is_none());
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = PrefixCache::new(2);

        let prefix_a: Vec<u32> = (0..32).collect();
        let prefix_b: Vec<u32> = (100..132).collect();
        let prefix_c: Vec<u32> = (200..232).collect();

        cache.store(prefix_a, make_dummy_cache(4));
        cache.store(prefix_b, make_dummy_cache(4));
        assert_eq!(cache.len(), 2);

        // Adding a third should evict the LRU (prefix_a, since it was stored first)
        cache.store(prefix_c.clone(), make_dummy_cache(4));
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
        cache.store(prefix.clone(), make_dummy_cache(4));
        assert!(cache.is_empty());

        let mut query = prefix;
        query.push(999);
        assert!(cache.find_longest_prefix(&query).is_none());
    }

    #[test]
    fn test_clear() {
        let mut cache = PrefixCache::new(10);
        let prefix: Vec<u32> = (0..32).collect();
        cache.store(prefix, make_dummy_cache(4));
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_longest_prefix_wins() {
        let mut cache = PrefixCache::new(10);

        // Store a shorter prefix
        let short_prefix: Vec<u32> = (0..20).collect();
        cache.store(short_prefix, make_dummy_cache(4));

        // Store a longer prefix that extends the short one
        let long_prefix: Vec<u32> = (0..50).collect();
        cache.store(long_prefix, make_dummy_cache(4));

        // Query should match the longer prefix
        let query: Vec<u32> = (0..64).collect();
        let result = cache.find_longest_prefix(&query);
        assert!(result.is_some());
        assert_eq!(result.unwrap().prefix_len, 50);
    }

    #[test]
    fn test_store_same_tokens_overwrites() {
        let mut cache = PrefixCache::new(10);
        let prefix: Vec<u32> = (0..32).collect();

        cache.store(prefix.clone(), make_dummy_cache(2));
        assert_eq!(cache.len(), 1);

        // Store again with same tokens but different cache layer count.
        // This is NOT a collision (same hash, same tokens), so it should overwrite.
        cache.store(prefix.clone(), make_dummy_cache(8));
        assert_eq!(cache.len(), 1);

        let mut query = prefix;
        query.push(999);
        let result = cache.find_longest_prefix(&query).unwrap();
        // The cache should reflect the second store (8 layers)
        assert_eq!(cache_layer_count(&result.cache), 8);
    }

    #[test]
    fn test_overwrite_does_not_trigger_eviction() {
        // With capacity 1, storing the same prefix twice should not evict anything
        // because the key already exists and the `contains_key` guard prevents eviction.
        let mut cache = PrefixCache::new(1);
        let prefix: Vec<u32> = (0..32).collect();

        cache.store(prefix.clone(), make_dummy_cache(2));
        assert_eq!(cache.len(), 1);

        // Overwrite with same prefix -- should NOT evict (key already present)
        cache.store(prefix.clone(), make_dummy_cache(4));
        assert_eq!(cache.len(), 1);

        let mut query = prefix;
        query.push(999);
        let result = cache.find_longest_prefix(&query).unwrap();
        assert_eq!(cache_layer_count(&result.cache), 4);
    }

    #[test]
    fn test_collision_guard_skips_different_tokens_same_hash() {
        let mut cache = PrefixCache::new(10);

        let prefix_a: Vec<u32> = (0..32).collect();
        let prefix_b: Vec<u32> = (100..132).collect();

        cache.store(prefix_a.clone(), make_dummy_cache(2));
        cache.store(prefix_b.clone(), make_dummy_cache(4));

        // Both should be stored since they have different hashes
        assert_eq!(cache.len(), 2);

        let mut query_a = prefix_a;
        query_a.push(999);
        let result_a = cache.find_longest_prefix(&query_a).unwrap();
        assert_eq!(cache_layer_count(&result_a.cache), 2);

        let mut query_b = prefix_b;
        query_b.push(999);
        let result_b = cache.find_longest_prefix(&query_b).unwrap();
        assert_eq!(cache_layer_count(&result_b.cache), 4);
    }

    #[test]
    fn test_eviction_only_when_new_key() {
        let mut cache = PrefixCache::new(2);

        let prefix_a: Vec<u32> = (0..32).collect();
        let prefix_b: Vec<u32> = (100..132).collect();
        let prefix_c: Vec<u32> = (200..232).collect();

        cache.store(prefix_a.clone(), make_dummy_cache(1));
        cache.store(prefix_b.clone(), make_dummy_cache(2));
        assert_eq!(cache.len(), 2);

        // Overwrite prefix_a: no eviction, still 2 entries
        cache.store(prefix_a.clone(), make_dummy_cache(3));
        assert_eq!(cache.len(), 2);

        // Verify both original keys still present
        let mut query_a = prefix_a;
        query_a.push(999);
        assert!(cache.find_longest_prefix(&query_a).is_some());

        let mut query_b = prefix_b;
        query_b.push(999);
        assert!(cache.find_longest_prefix(&query_b).is_some());

        // Now add a truly new prefix_c: should trigger eviction
        cache.store(prefix_c.clone(), make_dummy_cache(4));
        assert_eq!(cache.len(), 2);

        let mut query_c = prefix_c;
        query_c.push(999);
        assert!(cache.find_longest_prefix(&query_c).is_some());
    }
}
