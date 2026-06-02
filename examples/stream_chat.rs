//! Streaming chat with a local model and no tools, printing each text chunk as it arrives.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example stream_chat --features metal --release
//! ```

#[cfg(not(feature = "llama-cpp"))]
fn main() {
    eprintln!("Build with --features llama-cpp (or cuda / metal) to run this example.");
}

#[cfg(feature = "llama-cpp")]
#[tokio::main(flavor = "current_thread")]
async fn main() -> agents_rs::Result<()> {
    use std::io::Write;

    use agents_rs::{Chunk, Context, LocalAgent, LocalConfig, StreamingAgent, ToolEvent};
    use tokio_stream::StreamExt;
    use tracing_subscriber::EnvFilter;

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("agents_rs=info")),
        )
        .with_target(false)
        .init();

    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "models/NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf".into());
    let agent = LocalAgent::from_config(
        LocalConfig::new(model_path)
            .with_max_tokens(256)
            .with_temperature(0.2)
            .with_warmup(false),
    )?;

    let ctx = Context::new()
        .with_system("You are a concise assistant. Answer in 1-2 sentences.")
        .with_user("In one paragraph, write about Paris, France");

    let mut stream = agent.run_stream(ctx);
    print!("assistant: ");
    std::io::stdout().flush().ok();

    while let Some(item) = stream.next().await {
        match item? {
            Chunk::Text(s) => {
                print!("{s}");
                std::io::stdout().flush().ok();
            }
            Chunk::Tool(ToolEvent::Started { id, name }) => {
                println!("\n[tool start id={id} name={name}]");
            }
            Chunk::Tool(ToolEvent::Arguments { id, fragment }) => {
                println!("[tool args id={id} fragment={fragment:?}]");
            }
            Chunk::Tool(ToolEvent::Finished { id }) => {
                println!("[tool end id={id}]");
            }
            Chunk::Usage(u) => {
                eprintln!("\n[usage {u:?}]");
            }
            Chunk::End => {
                println!();
                break;
            }
        }
    }
    Ok(())
}
