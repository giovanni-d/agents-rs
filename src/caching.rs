//! In-memory caching wrapper for any [`Agent`], keyed by [`Context`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::{Agent, BoxFuture, Context, Response, Result};

/// Memoizes an [`Agent`]'s responses keyed by [`Context`].
pub struct CachingAgent<A> {
    inner: A,
    cache: Arc<Mutex<HashMap<String, Response>>>,
}

impl<A> CachingAgent<A> {
    pub fn new(inner: A) -> Self {
        Self {
            inner,
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Drops every cached entry.
    pub fn clear(&self) {
        if let Ok(mut g) = self.cache.lock() {
            g.clear();
        }
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.cache.lock().map(|g| g.len()).unwrap_or(0)
    }
}

impl<A: Agent + 'static> Agent for CachingAgent<A> {
    fn run<'a>(&'a self, ctx: Context) -> BoxFuture<'a, Result<Response>> {
        let key = match cache_key(&ctx) {
            Ok(k) => k,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        if let Ok(g) = self.cache.lock()
            && let Some(hit) = g.get(&key).cloned()
        {
            tracing::debug!(key_bytes = key.len(), "cache hit");
            return Box::pin(async move { Ok(hit) });
        }
        let cache = Arc::clone(&self.cache);
        let fut = self.inner.run(ctx);
        Box::pin(async move {
            let response = fut.await?;
            if let Ok(mut g) = cache.lock() {
                g.insert(key, response.clone());
            }
            Ok(response)
        })
    }
}

/// Stable key from serialized messages plus the optional structured-output request.
fn cache_key(ctx: &Context) -> Result<String> {
    #[derive(serde::Serialize)]
    struct Key<'a> {
        messages: &'a [crate::Message],
        schema: Option<&'a crate::OutputSchemaRequest>,
    }
    let k = Key {
        messages: &ctx.messages,
        schema: ctx.output_schema_request(),
    };
    Ok(serde_json::to_string(&k)?)
}
