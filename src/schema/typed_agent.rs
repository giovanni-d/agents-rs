//! [`TypedAgent`] with compile-time input/output types, plus a [`TypedAgentAdapter`]
//! that bridges to the untyped [`Agent`] trait by validating JSON in and out.

use std::future::Future;

use super::AgentData;
use crate::agent::{Agent, Response};
use crate::error::{AgentError, BoxFuture, Result};
use crate::message::Context;

/// An agent whose input and output types are fixed at compile time.
pub trait TypedAgent: Send + Sync {
    type Input: AgentData;
    type Output: AgentData;
    type Future: Future<Output = Result<Self::Output>> + Send;

    fn run(&self, input: Self::Input) -> Self::Future;
}

/// Wraps a [`TypedAgent`] so it satisfies the untyped [`Agent`] contract.
///
/// On each call the adapter parses the last user message as JSON, validates
/// it against `A::Input::schema()`, calls the inner agent, then serializes
/// the output back into [`Response::Message`].
pub struct TypedAgentAdapter<A: TypedAgent> {
    inner: A,
}

impl<A: TypedAgent> TypedAgentAdapter<A> {
    pub fn new(agent: A) -> Self {
        Self { inner: agent }
    }

    pub fn inner(&self) -> &A {
        &self.inner
    }

    pub fn into_inner(self) -> A {
        self.inner
    }
}

impl<A: TypedAgent + 'static> Agent for TypedAgentAdapter<A> {
    fn run<'a>(&'a self, ctx: Context) -> BoxFuture<'a, Result<Response>> {
        let input = match extract_input::<A::Input>(&ctx) {
            Ok(v) => v,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        let fut = self.inner.run(input);
        Box::pin(async move {
            let output = fut.await?;
            let json = serde_json::to_string(&output)?;
            Ok(Response::Message(json))
        })
    }
}

fn extract_input<T: AgentData>(ctx: &Context) -> Result<T> {
    let msg = ctx
        .last_user_message()
        .ok_or_else(|| AgentError::Other("no user message to extract typed input from".into()))?;
    let value: serde_json::Value = serde_json::from_str(msg)?;
    T::schema()
        .validate(&value)
        .map_err(|e| AgentError::Other(format!("schema validation failed: {e}")))?;
    Ok(serde_json::from_value(value)?)
}

/// Convenience: `agent.into_agent()` wraps any [`TypedAgent`] in a [`TypedAgentAdapter`].
pub trait TypedAgentExt: TypedAgent + Sized {
    fn into_agent(self) -> TypedAgentAdapter<Self>
    where
        Self: 'static,
    {
        TypedAgentAdapter::new(self)
    }
}

impl<A: TypedAgent> TypedAgentExt for A {}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::pin::Pin;

    use crate::schema::SchemaKind;

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct TestInput {
        value: i32,
    }
    impl AgentData for TestInput {
        fn schema() -> SchemaKind {
            SchemaKind::object()
                .field("value", SchemaKind::integer())
                .build()
        }
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct TestOutput {
        result: i32,
    }
    impl AgentData for TestOutput {
        fn schema() -> SchemaKind {
            SchemaKind::object()
                .field("result", SchemaKind::integer())
                .build()
        }
    }

    struct DoubleAgent;
    impl TypedAgent for DoubleAgent {
        type Input = TestInput;
        type Output = TestOutput;
        type Future = Pin<Box<dyn Future<Output = Result<TestOutput>> + Send>>;
        fn run(&self, input: TestInput) -> Self::Future {
            Box::pin(async move {
                Ok(TestOutput {
                    result: input.value * 2,
                })
            })
        }
    }

    #[tokio::test]
    async fn direct_invocation_runs_typed_logic() {
        let agent = DoubleAgent;
        let out = agent.run(TestInput { value: 21 }).await.unwrap();
        assert_eq!(out.result, 42);
    }

    #[tokio::test]
    async fn adapter_parses_json_input_and_serializes_output() {
        let agent = DoubleAgent.into_agent();
        let ctx = Context::new().with_user(r#"{"value": 21}"#);
        let Response::Message(s) = agent.run(ctx).await.unwrap() else {
            panic!("expected message response");
        };
        let parsed: TestOutput = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed.result, 42);
    }

    #[tokio::test]
    async fn adapter_rejects_input_that_fails_schema_validation() {
        let agent = DoubleAgent.into_agent();
        let ctx = Context::new().with_user(r#"{"value": "not a number"}"#);
        assert!(agent.run(ctx).await.is_err());
    }

    #[tokio::test]
    async fn adapter_rejects_input_missing_required_field() {
        let agent = DoubleAgent.into_agent();
        let ctx = Context::new().with_user(r#"{}"#);
        assert!(agent.run(ctx).await.is_err());
    }

    #[tokio::test]
    async fn adapter_errors_without_user_message() {
        let agent = DoubleAgent.into_agent();
        let ctx = Context::new();
        assert!(agent.run(ctx).await.is_err());
    }
}
