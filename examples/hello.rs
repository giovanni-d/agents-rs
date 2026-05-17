//! End-to-end demo: a local GGUF model invokes a tool and returns the result.
//! Logging is opt-in via a `tracing` subscriber — set `RUST_LOG=agents_rs=debug`
//! to see tool-call args, results, and per-iteration timing.
//!
//! Run with:
//!
//! ```sh
//! RUST_LOG=agents_rs=info cargo run --example hello --features llama-cpp --release
//! ```

#[cfg(not(feature = "llama-cpp"))]
fn main() {
    eprintln!("Build with --features llama-cpp to run this example.");
}

#[cfg(feature = "llama-cpp")]
#[tokio::main(flavor = "current_thread")]
async fn main() -> agents_rs::Result<()> {
    use agents_rs::{
        Context, FnTool, LocalAgent, LocalConfig, LoggingAgent, ToolRegistry, run_with_tools,
    };
    use serde_json::json;
    use tracing_subscriber::EnvFilter;

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("agents_rs=info")),
        )
        .with_target(false)
        .init();

    let add = FnTool::new(
        "add",
        "Add two integers and return their sum.",
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "integer" },
                "b": { "type": "integer" }
            },
            "required": ["a", "b"]
        }),
        |args| async move {
            let a = args["a"].as_i64().unwrap_or(0);
            let b = args["b"].as_i64().unwrap_or(0);
            Ok(json!({ "result": a + b }))
        },
    );
    let registry = ToolRegistry::new().register(add);

    let local = LocalAgent::from_config(
        LocalConfig::new("models/NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf")
            .with_max_tokens(128)
            .with_temperature(0.2),
    )?
    .with_tools(&registry);

    let system = "You are a helpful assistant. Use the available tools when appropriate.";
    local.prime_for_system(system).await?;

    let agent = LoggingAgent::new("local", local);

    let ctx = Context::new()
        .with_system(system)
        .with_user("What is 12 + 30? Use the add tool to compute it.");

    let answer = run_with_tools(&agent, &registry, ctx, 4).await?;
    println!("\nFinal answer:\n{answer}");
    Ok(())
}
