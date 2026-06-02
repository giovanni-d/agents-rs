//! The [`Agent`] trait, its [`Response`] shape, and an [`FnAgent`] closure adapter.

use std::future::Future;

use serde::{Deserialize, Serialize};

use crate::error::{BoxFuture, Result};
use crate::message::Context;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone)]
pub enum Response {
    Message(String),
    ToolCalls(Vec<ToolCall>),
}

impl Response {
    pub fn message(s: impl Into<String>) -> Self {
        Response::Message(s.into())
    }
    pub fn tool_calls(calls: Vec<ToolCall>) -> Self {
        Response::ToolCalls(calls)
    }
}

pub trait Agent: Send + Sync {
    fn run<'a>(&'a self, ctx: Context) -> BoxFuture<'a, Result<Response>>;
}

pub struct FnAgent<F>(F);

pub fn fn_agent<F, Fut>(f: F) -> FnAgent<F>
where
    F: Fn(Context) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Response>> + Send + 'static,
{
    FnAgent(f)
}

impl<F, Fut> Agent for FnAgent<F>
where
    F: Fn(Context) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Response>> + Send + 'static,
{
    fn run<'a>(&'a self, ctx: Context) -> BoxFuture<'a, Result<Response>> {
        Box::pin((self.0)(ctx))
    }
}
