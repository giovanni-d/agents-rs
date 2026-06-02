//! Typed tools, type-erased [`DynTool`] handles, and the [`ToolRegistry`] that dispatches them.

use std::collections::HashMap;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::error::{AgentError, Result};
use crate::schema::SchemaKind;

/// Static metadata for a [`Tool`]: name, description, and input/output schemas.
#[derive(Clone, Debug)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input: SchemaKind,
    pub output: Option<SchemaKind>,
}

impl ToolDefinition {
    pub fn builder(
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> ToolDefinitionBuilder {
        ToolDefinitionBuilder {
            name: name.into(),
            description: description.into(),
            input: None,
            output: None,
        }
    }

    /// Anthropic `tools` array entry.
    pub fn to_anthropic_tool(&self) -> Value {
        serde_json::json!({
            "name": self.name,
            "description": self.description,
            "input_schema": self.input.to_json_schema(),
        })
    }
}

pub struct ToolDefinitionBuilder {
    name: String,
    description: String,
    input: Option<SchemaKind>,
    output: Option<SchemaKind>,
}

impl ToolDefinitionBuilder {
    pub fn input(mut self, schema: impl Into<SchemaKind>) -> Self {
        self.input = Some(schema.into());
        self
    }

    pub fn output(mut self, schema: impl Into<SchemaKind>) -> Self {
        self.output = Some(schema.into());
        self
    }

    pub fn build(self) -> ToolDefinition {
        ToolDefinition {
            name: self.name,
            description: self.description,
            input: self.input.unwrap_or(SchemaKind::Object { fields: vec![] }),
            output: self.output,
        }
    }
}

/// Anything a [`Tool`] can return; blanket-implemented for `Serialize + Send + Sync`.
pub trait ToolOutput: Send + Sync {
    fn to_json(&self) -> Result<Value>;
}

impl<T: Serialize + Send + Sync> ToolOutput for T {
    fn to_json(&self) -> Result<Value> {
        serde_json::to_value(self).map_err(Into::into)
    }
}

/// A typed tool: JSON arguments in, JSON-serializable output.
pub trait Tool: Send + Sync {
    type Input: DeserializeOwned + Send;
    type Output: ToolOutput;
    type Future: Future<Output = Result<Self::Output>> + Send;

    fn definition(&self) -> ToolDefinition;
    fn execute(&self, input: Self::Input) -> Self::Future;
}

/// Hides [`Tool`] associated types behind a uniform `Value`-in / `Value`-out signature.
trait ErasedTool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    fn execute_erased<'a>(
        &'a self,
        input: Value,
    ) -> Pin<Box<dyn Future<Output = Result<Value>> + Send + 'a>>;
}

impl<T: Tool> ErasedTool for T {
    fn definition(&self) -> ToolDefinition {
        Tool::definition(self)
    }

    fn execute_erased<'a>(
        &'a self,
        input: Value,
    ) -> Pin<Box<dyn Future<Output = Result<Value>> + Send + 'a>> {
        Box::pin(async move {
            let typed: T::Input = serde_json::from_value(input).map_err(|e| {
                AgentError::Other(format!("tool input parse failed: {e}"))
            })?;
            let output = self.execute(typed).await?;
            output.to_json()
        })
    }
}

/// Cheaply cloneable, type-erased handle to a registered tool.
#[derive(Clone)]
pub struct DynTool(Arc<dyn ErasedTool>);

impl DynTool {
    pub fn new<T: Tool + 'static>(tool: T) -> Self {
        DynTool(Arc::new(tool))
    }

    pub fn definition(&self) -> ToolDefinition {
        self.0.definition()
    }

    pub fn name(&self) -> String {
        self.definition().name
    }

    pub async fn execute(&self, arguments: Value) -> Result<Value> {
        self.0.execute_erased(arguments).await
    }
}

impl std::fmt::Debug for DynTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynTool")
            .field("name", &self.definition().name)
            .finish()
    }
}

/// Closure-backed [`Tool`].
pub struct FnTool<F, In, Out, Fut>
where
    F: Fn(In) -> Fut + Send + Sync,
    In: DeserializeOwned + Send,
    Out: ToolOutput,
    Fut: Future<Output = Result<Out>> + Send,
{
    definition: ToolDefinition,
    func: F,
    _marker: PhantomData<fn(In) -> (Out, Fut)>,
}

impl<F, In, Out, Fut> FnTool<F, In, Out, Fut>
where
    F: Fn(In) -> Fut + Send + Sync,
    In: DeserializeOwned + Send,
    Out: ToolOutput,
    Fut: Future<Output = Result<Out>> + Send,
{
    pub fn new(definition: ToolDefinition, func: F) -> Self {
        Self {
            definition,
            func,
            _marker: PhantomData,
        }
    }
}

impl<F, In, Out, Fut> Tool for FnTool<F, In, Out, Fut>
where
    F: Fn(In) -> Fut + Send + Sync,
    In: DeserializeOwned + Send,
    Out: ToolOutput + 'static,
    Fut: Future<Output = Result<Out>> + Send + 'static,
{
    type Input = In;
    type Output = Out;
    type Future = Fut;

    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    fn execute(&self, input: Self::Input) -> Self::Future {
        (self.func)(input)
    }
}

/// Convenience constructor for [`FnTool`].
pub fn fn_tool<F, In, Out, Fut>(
    definition: ToolDefinition,
    func: F,
) -> FnTool<F, In, Out, Fut>
where
    F: Fn(In) -> Fut + Send + Sync,
    In: DeserializeOwned + Send,
    Out: ToolOutput,
    Fut: Future<Output = Result<Out>> + Send,
{
    FnTool::new(definition, func)
}

/// Name-keyed registry of tools with combined-schema and GBNF-grammar exports.
#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, DynTool>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<T: Tool + 'static>(mut self, tool: T) -> Self {
        let dyn_tool = DynTool::new(tool);
        self.tools.insert(dyn_tool.name(), dyn_tool);
        self
    }

    pub fn register_dyn(mut self, tool: DynTool) -> Self {
        self.tools.insert(tool.name(), tool);
        self
    }

    pub fn get(&self, name: &str) -> Option<&DynTool> {
        self.tools.get(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tools.keys().map(|s| s.as_str())
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|t| t.definition()).collect()
    }

    pub fn tools(&self) -> impl Iterator<Item = &DynTool> {
        self.tools.values()
    }

    /// Execute a registered tool by name.
    ///
    /// # Errors
    /// Returns [`AgentError::UnknownTool`] if the name isn't registered.
    pub async fn execute(&self, name: &str, arguments: Value) -> Result<Value> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| AgentError::UnknownTool(name.to_string()))?;
        tool.execute(arguments).await
    }

    /// Anthropic-style `tools` array.
    pub fn to_anthropic_tools(&self) -> Vec<Value> {
        self.tools
            .values()
            .map(|t| t.definition().to_anthropic_tool())
            .collect()
    }

    /// Enum schema with one variant per tool, payload being the tool's input schema.
    pub fn to_combined_schema(&self) -> SchemaKind {
        let mut builder = SchemaKind::enumeration();
        for tool in self.tools.values() {
            let def = tool.definition();
            builder = builder.variant(def.name, def.input);
        }
        builder.build()
    }

    /// GBNF grammar matching any tool-call envelope `{"name":"<tool>","arguments":<schema>}`.
    pub fn to_tool_call_grammar(&self) -> String {
        if self.tools.is_empty() {
            return String::from("root ::= \"{}\"");
        }
        let alts: Vec<String> = self
            .tools
            .values()
            .map(|tool| {
                let def = tool.definition();
                format!(
                    "(\"{{\\\"name\\\":\\\"{}\\\"\" \",\\\"arguments\\\":\" {} \"}}\")",
                    def.name,
                    def.input.to_gbnf()
                )
            })
            .collect();
        format!("root ::= {}", alts.join(" | "))
    }

    pub fn merge(mut self, other: ToolRegistry) -> Self {
        self.tools.extend(other.tools);
        self
    }
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl FromIterator<DynTool> for ToolRegistry {
    fn from_iter<I: IntoIterator<Item = DynTool>>(iter: I) -> Self {
        let mut reg = ToolRegistry::new();
        for tool in iter {
            reg = reg.register_dyn(tool);
        }
        reg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::future::{Ready, ready};

    #[derive(Deserialize)]
    struct EchoInput {
        message: String,
    }

    struct EchoTool;
    impl Tool for EchoTool {
        type Input = EchoInput;
        type Output = String;
        type Future = Ready<Result<String>>;
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::builder("echo", "Echo back the message")
                .input(SchemaKind::object().field("message", SchemaKind::string()))
                .output(SchemaKind::string())
                .build()
        }
        fn execute(&self, input: Self::Input) -> Self::Future {
            ready(Ok(input.message))
        }
    }

    #[derive(Deserialize)]
    struct AddInput {
        a: f64,
        b: f64,
    }

    struct AddTool;
    impl Tool for AddTool {
        type Input = AddInput;
        type Output = f64;
        type Future = Ready<Result<f64>>;
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::builder("add", "Add two numbers")
                .input(
                    SchemaKind::object()
                        .field("a", SchemaKind::number())
                        .field("b", SchemaKind::number()),
                )
                .output(SchemaKind::number())
                .build()
        }
        fn execute(&self, input: Self::Input) -> Self::Future {
            ready(Ok(input.a + input.b))
        }
    }

    #[test]
    fn register_and_lookup() {
        let registry = ToolRegistry::new().register(EchoTool).register(AddTool);
        assert_eq!(registry.len(), 2);
        assert!(registry.contains("echo"));
        assert!(registry.contains("add"));
        assert!(registry.get("missing").is_none());
    }

    #[tokio::test]
    async fn execute_typed_tool_via_json() {
        let registry = ToolRegistry::new().register(AddTool);
        let out = registry
            .execute("add", serde_json::json!({"a": 2.0, "b": 3.0}))
            .await
            .unwrap();
        assert_eq!(out, serde_json::json!(5.0));
    }

    #[tokio::test]
    async fn execute_unknown_tool_yields_unknown_tool_error() {
        let registry = ToolRegistry::new();
        let err = registry
            .execute("missing", serde_json::json!({}))
            .await
            .unwrap_err();
        assert!(matches!(err, AgentError::UnknownTool(_)));
    }

    #[tokio::test]
    async fn execute_with_malformed_input_errors() {
        let registry = ToolRegistry::new().register(AddTool);
        let err = registry
            .execute("add", serde_json::json!({"a": "nope"}))
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("tool input parse failed"));
    }

    #[tokio::test]
    async fn fn_tool_runs_closure_and_returns_value() {
        let tool = FnTool::new(
            ToolDefinition::builder("double", "Multiply by 2")
                .input(SchemaKind::object().field("x", SchemaKind::number()))
                .output(SchemaKind::number())
                .build(),
            |args: serde_json::Value| async move {
                let x = args["x"].as_f64().unwrap();
                Ok(x * 2.0)
            },
        );
        let registry = ToolRegistry::new().register(tool);
        let out = registry
            .execute("double", serde_json::json!({"x": 21.0}))
            .await
            .unwrap();
        assert_eq!(out, serde_json::json!(42.0));
    }

    #[test]
    fn combined_schema_has_one_variant_per_tool() {
        let registry = ToolRegistry::new().register(EchoTool).register(AddTool);
        let SchemaKind::Enum { variants } = registry.to_combined_schema() else {
            panic!("expected enum schema");
        };
        assert_eq!(variants.len(), 2);
    }

    #[test]
    fn anthropic_tools_carry_input_schema() {
        let registry = ToolRegistry::new().register(EchoTool);
        let tools = registry.to_anthropic_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "echo");
        assert_eq!(tools[0]["input_schema"]["type"], "object");
    }

    #[test]
    fn tool_call_grammar_lists_each_tool_as_alternative() {
        let registry = ToolRegistry::new().register(EchoTool).register(AddTool);
        let grammar = registry.to_tool_call_grammar();
        assert!(grammar.starts_with("root ::="));
        assert!(grammar.contains("echo"));
        assert!(grammar.contains("add"));
    }

    #[test]
    fn empty_registry_grammar_is_empty_object() {
        let grammar = ToolRegistry::new().to_tool_call_grammar();
        assert_eq!(grammar, "root ::= \"{}\"");
    }
}
