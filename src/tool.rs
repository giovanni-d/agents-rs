//! [`Tool`] trait, an [`FnTool`] closure adapter, and a [`ToolRegistry`].

use std::future::Future;
use std::sync::Arc;

use crate::error::{BoxFuture, Result};

pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;
    fn call<'a>(&'a self, args: serde_json::Value) -> BoxFuture<'a, Result<serde_json::Value>>;
}

type ToolFn =
    dyn Fn(serde_json::Value) -> BoxFuture<'static, Result<serde_json::Value>> + Send + Sync;

pub struct FnTool {
    name: String,
    description: String,
    schema: serde_json::Value,
    func: Arc<ToolFn>,
}

impl FnTool {
    pub fn new<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        schema: serde_json::Value,
        func: F,
    ) -> Self
    where
        F: Fn(serde_json::Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<serde_json::Value>> + Send + 'static,
    {
        Self {
            name: name.into(),
            description: description.into(),
            schema,
            func: Arc::new(move |args| Box::pin(func(args))),
        }
    }
}

impl Tool for FnTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn parameters_schema(&self) -> serde_json::Value {
        self.schema.clone()
    }
    fn call<'a>(&'a self, args: serde_json::Value) -> BoxFuture<'a, Result<serde_json::Value>> {
        (self.func)(args)
    }
}

#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: Vec<Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.push(Arc::new(tool));
        self
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.iter().find(|t| t.name() == name)
    }

    pub fn list(&self) -> &[Arc<dyn Tool>] {
        &self.tools
    }
}
