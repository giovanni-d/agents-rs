#![allow(dead_code)]

use agents_rs::{
    AgentError, FnTool, LocalConfig, SchemaKind, ToolDefinition, ToolRegistry,
    templates::{
        ChatMLTemplate, ChatTemplateKind, Gemma4Template, Llama3Template, MistralTemplate,
        ThinkingMode,
    },
};
use serde::{Deserialize, Serialize};
use tracing_subscriber::EnvFilter;

#[derive(Deserialize)]
pub struct TwoNumbers {
    pub a: f64,
    pub b: f64,
}

#[derive(Serialize)]
pub struct OpResult {
    pub result: f64,
}

#[derive(Debug, Deserialize)]
pub struct FinalAnswer {
    pub answer: f64,
}

pub fn two_numbers_schema() -> SchemaKind {
    SchemaKind::object()
        .field_with_desc("a", SchemaKind::number(), "Left operand")
        .field_with_desc("b", SchemaKind::number(), "Right operand")
        .build()
}

pub fn op_result_schema() -> SchemaKind {
    SchemaKind::object()
        .field_with_desc("result", SchemaKind::number(), "Operation result")
        .build()
}

pub fn final_answer_schema() -> SchemaKind {
    SchemaKind::object()
        .field_with_desc("answer", SchemaKind::number(), "The computed result")
        .build()
}

pub fn build_calc_registry() -> ToolRegistry {
    let two = two_numbers_schema();
    let out = op_result_schema();
    let def = |name: &str, desc: &str| -> ToolDefinition {
        ToolDefinition::builder(name, desc)
            .input(two.clone())
            .output(out.clone())
            .build()
    };
    let add = FnTool::new(
        def("add", "Add two numbers and return their sum."),
        |TwoNumbers { a, b }| async move { Ok(OpResult { result: a + b }) },
    );
    let subtract = FnTool::new(
        def("subtract", "Subtract b from a and return the difference (a - b)."),
        |TwoNumbers { a, b }| async move { Ok(OpResult { result: a - b }) },
    );
    let multiply = FnTool::new(
        def("multiply", "Multiply two numbers and return their product."),
        |TwoNumbers { a, b }| async move { Ok(OpResult { result: a * b }) },
    );
    let divide = FnTool::new(
        def("divide", "Divide a by b. Errors if b is zero."),
        |TwoNumbers { a, b }| async move {
            if b == 0.0 {
                return Err(AgentError::Other("division by zero".into()));
            }
            Ok(OpResult { result: a / b })
        },
    );
    ToolRegistry::new()
        .register(add)
        .register(subtract)
        .register(multiply)
        .register(divide)
}

pub fn build_calc_system_prompt(final_schema: &SchemaKind) -> String {
    let final_schema_json = final_schema.to_json_schema();
    format!(
        "You are a math assistant. You MUST use the provided tools for \
         every single arithmetic operation. Do not compute any number in \
         your head; every `+`, `-`, `*`, `/` must go through a tool call.\n\
         \n\
         Rules:\n\
         1. Call exactly ONE tool per turn. After the tool returns, wait \
            for the next turn before calling another tool.\n\
         2. Pick the tool that executes the NEXT operator in the original \
            expression, following standard arithmetic precedence \
            (parentheses, then `*`/`/`, then `+`/`-`).\n\
         3. Substitute the actual number — for example `12` or `5.0` — \
            into the tool's argument. NEVER pass the literal text \
            \"result\" or any other word as a number; arguments must \
            always be JSON numbers.\n\
         4. After every tool result, re-read the ORIGINAL expression. If \
            any operator has not yet gone through a tool call, you MUST \
            call the next tool — do not finalize the answer. It is \
            FORBIDDEN to compute any step in your head.\n\
         5. Only when every operator in the original expression has been \
            performed via a tool may you reply with the final numeric \
            answer as plain text.\n\
         \n\
         The host will parse your final answer into a JSON object matching \
         this schema:\n{}",
        serde_json::to_string(&final_schema_json).expect("schema is serializable"),
    )
}

pub const CALC_USER_PROMPT: &str =
    "Compute (8 + 4) * 3 - (10 / 2). Reply with the JSON answer object.";

pub fn model_path_from_env() -> String {
    std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "models/NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf".into())
}

/// Map a GGUF filename to a chat-template override, or `None` to keep llama.cpp's default.
pub fn auto_template_for(path: &str) -> Option<ChatTemplateKind> {
    let lower = path.to_lowercase();
    if lower.contains("llama-3.1") || lower.contains("llama-3.2") {
        Some(ChatTemplateKind::Llama3(Llama3Template))
    } else if lower.contains("gemma") {
        Some(ChatTemplateKind::Gemma4(Gemma4Template { thinking: false }))
    } else if lower.contains("ministral") || lower.contains("mistral") {
        Some(ChatTemplateKind::Mistral(MistralTemplate))
    } else if lower.contains("qwen") {
        Some(ChatTemplateKind::ChatML(ChatMLTemplate {
            thinking: ThinkingMode::Suppressed,
        }))
    } else {
        None
    }
}

pub fn build_calc_config(model_path: &str) -> LocalConfig {
    let mut config = LocalConfig::new(model_path)
        .with_max_tokens(256)
        .with_temperature(0.0);
    if let Some(t) = auto_template_for(model_path) {
        config = config.with_chat_template(t);
    }
    config
}

pub fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("agents_rs=info")),
        )
        .with_target(false)
        .init();
}
