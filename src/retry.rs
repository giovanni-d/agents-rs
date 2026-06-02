//! Retry-on-error wrapper for any [`Agent`]. Only retries `Err` results, not semantic failures.

use std::time::Duration;

use crate::{Agent, BoxFuture, Context, Response, Result};

/// Wraps an [`Agent`] and retries `inner.run(ctx)` on `Err`.
pub struct RetryAgent<A> {
    inner: A,
    max_retries: u32,
    delay: Duration,
}

impl<A> RetryAgent<A> {
    pub fn new(inner: A, max_retries: u32) -> Self {
        Self {
            inner,
            max_retries,
            delay: Duration::from_millis(0),
        }
    }

    /// Fixed delay between attempts.
    pub fn with_delay_ms(mut self, ms: u64) -> Self {
        self.delay = Duration::from_millis(ms);
        self
    }
}

impl<A: Agent + 'static> Agent for RetryAgent<A> {
    fn run<'a>(&'a self, ctx: Context) -> BoxFuture<'a, Result<Response>> {
        let max = self.max_retries;
        let delay = self.delay;
        Box::pin(async move {
            let mut attempts: u32 = 0;
            loop {
                match self.inner.run(ctx.clone()).await {
                    Ok(r) => {
                        if attempts > 0 {
                            tracing::info!(attempts, "retry: succeeded after retries");
                        }
                        return Ok(r);
                    }
                    Err(e) => {
                        if attempts >= max {
                            return Err(e);
                        }
                        tracing::warn!(
                            attempt = attempts + 1,
                            max,
                            error = %e,
                            "retry: agent.run failed, retrying"
                        );
                        attempts += 1;
                        if !delay.is_zero() {
                            tokio::time::sleep(delay).await;
                        }
                    }
                }
            }
        })
    }
}
