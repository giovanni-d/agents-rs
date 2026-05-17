//! Multi-tool calculator: a local GGUF model decomposes an arithmetic
//! expression into atomic operations, calls the appropriate tool for each
//! step, and threads the result of each call into the next.
//!
//! Demonstrates:
//! - [`LocalAgent`] driven by llama.cpp.
//! - Several tools registered on a single agent.
//! - Multi-iteration reasoning: the model passes one tool's output as the
//!   input of the next, then answers the user.
//! - KV-cache prefix priming for the static system prompt.
//! - `tracing`-based logging of every loop iteration and agent call.
//!
//! Run with:
//!
//! ```sh
//! RUST_LOG=agents_rs=info cargo run --example calculator --features cuda --release
//! ```

#[cfg(not(feature = "llama-cpp"))]
fn main() {
    eprintln!("Build with --features llama-cpp (or cuda / metal) to run this example.");
}

#[cfg(feature = "llama-cpp")]
#[tokio::main(flavor = "current_thread")]
async fn main() -> agents_rs::Result<()> {
    use agents_rs::{
        AgentError, Context, FnTool, LocalAgent, LocalConfig, LoggingAgent, ToolRegistry,
        run_with_tools,
    };
    use serde_json::{Value, json};
    use tracing_subscriber::EnvFilter;

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("agents_rs=info")),
        )
        .with_target(false)
        .init();

    fn pair(args: &Value) -> agents_rs::Result<(f64, f64)> {
        let a = args["a"]
            .as_f64()
            .ok_or_else(|| AgentError::Other("missing or non-numeric 'a'".into()))?;
        let b = args["b"]
            .as_f64()
            .ok_or_else(|| AgentError::Other("missing or non-numeric 'b'".into()))?;
        Ok((a, b))
    }

    let two_numbers = json!({
        "type": "object",
        "properties": {
            "a": { "type": "number" },
            "b": { "type": "number" }
        },
        "required": ["a", "b"]
    });

    let add = FnTool::new(
        "add",
        "Add two numbers and return their sum.",
        two_numbers.clone(),
        |args| async move {
            let (a, b) = pair(&args)?;
            Ok(json!({ "result": a + b }))
        },
    );
    let subtract = FnTool::new(
        "subtract",
        "Subtract b from a and return the difference (a - b).",
        two_numbers.clone(),
        |args| async move {
            let (a, b) = pair(&args)?;
            Ok(json!({ "result": a - b }))
        },
    );
    let multiply = FnTool::new(
        "multiply",
        "Multiply two numbers and return their product.",
        two_numbers.clone(),
        |args| async move {
            let (a, b) = pair(&args)?;
            Ok(json!({ "result": a * b }))
        },
    );
    let divide = FnTool::new(
        "divide",
        "Divide a by b. Errors if b is zero.",
        two_numbers,
        |args| async move {
            let (a, b) = pair(&args)?;
            if b == 0.0 {
                return Err(AgentError::Other("division by zero".into()));
            }
            Ok(json!({ "result": a / b }))
        },
    );

    let registry = ToolRegistry::new()
        .register(add)
        .register(subtract)
        .register(multiply)
        .register(divide);

    let local = LocalAgent::from_config(
        LocalConfig::new("models/NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf")
            .with_max_tokens(256)
            .with_temperature(0.2),
    )?
    .with_tools(&registry);

    let system = "You are a math assistant. Break each expression into atomic \
                  arithmetic steps and use the provided tools for every step. \
                  Pass the result of one tool call as input to the next. When \
                  all steps are computed, reply to the user with the final number.";
    local.prime_for_system(system).await?;

    let agent = LoggingAgent::new("local", local);

    let ctx = Context::new()
        .with_system(system)
        .with_user("Compute (8 + 4) * 3 - (10 / 2). Show only the final number.");

    let answer = run_with_tools(&agent, &registry, ctx, 10).await?;
    println!("\nFinal answer:\n{answer}");
    Ok(())
}
