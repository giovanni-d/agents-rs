//! Local-model [`Agent`] powered by llama.cpp via `llama-cpp-2`.
//!
//! A dedicated OS thread owns the loaded `LlamaModel` and `LlamaContext` so the
//! KV cache stays allocated across calls. Messages are formatted as ChatML
//! (works for Qwen / Nemotron-class instruct models). Tool calls are produced
//! by prompting the model to emit a small JSON envelope and parsing it back
//! out of the generated text — no grammar constraints.

use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::mpsc;

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

use crate::tool_parser::extract_tool_calls;
use crate::{Agent, AgentError, BoxFuture, Context, Message, Response, Result, ToolRegistry};

// ---------- Config ----------

#[derive(Debug, Clone)]
pub struct LocalConfig {
    pub model_path: String,
    pub n_ctx: u32,
    pub n_gpu_layers: u32,
    pub max_tokens: u32,
    pub temperature: f32,
    pub seed: u32,
    pub flash_attention: bool,
}

impl LocalConfig {
    pub fn new(model_path: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            n_ctx: 4096,
            n_gpu_layers: 99,
            max_tokens: 512,
            temperature: 0.3,
            seed: 42,
            flash_attention: true,
        }
    }

    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = n;
        self
    }

    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    pub fn with_n_ctx(mut self, n: u32) -> Self {
        self.n_ctx = n;
        self
    }

    pub fn with_n_gpu_layers(mut self, n: u32) -> Self {
        self.n_gpu_layers = n;
        self
    }

    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_flash_attention(mut self, enabled: bool) -> Self {
        self.flash_attention = enabled;
        self
    }
}

// ---------- KV cache sequence ids ----------
//
// Two sequences are allocated. PRIME holds the static prefix decoded once
// by `prime`; INFER does all inference and gets full-cleared on every call.
// Matching prompts copy PRIME's cells into INFER (no data move — llama.cpp
// shares them) and skip re-decoding the prefix.
const PRIME_SEQ_ID: i32 = 0;
const INFER_SEQ_ID: i32 = 1;

// ---------- Process-global backend ----------

static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

fn shared_backend() -> &'static LlamaBackend {
    BACKEND.get_or_init(|| {
        // Route llama.cpp's C-side logs through `tracing`, then disable them so
        // they don't show up unless a subscriber explicitly enables the
        // `llama_cpp_2` target. Called once per process — that's all the API
        // supports.
        llama_cpp_2::send_logs_to_tracing(
            llama_cpp_2::LogOptions::default().with_logs_enabled(false),
        );
        LlamaBackend::init().expect("llama backend init failed")
    })
}

// ---------- Agent ----------

enum Request {
    Infer {
        prompt: String,
        reply: tokio::sync::oneshot::Sender<Result<String>>,
    },
    Prime {
        prefix: String,
        reply: tokio::sync::oneshot::Sender<Result<usize>>,
    },
    Shutdown,
}

/// Agent that drives a local GGUF model.
///
/// Use [`Self::with_tools`] to attach a [`ToolRegistry`]; the agent then
/// injects a tool catalog into the system prompt and tries to parse a
/// `{"tool": "...", "args": {...}}` envelope out of each response.
pub struct LocalAgent {
    tx: Arc<mpsc::Sender<Request>>,
    handle: Option<std::thread::JoinHandle<()>>,
    tools_summary: Option<String>,
}

impl LocalAgent {
    pub fn from_config(config: LocalConfig) -> Result<Self> {
        let (tx, rx) = mpsc::channel::<Request>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<()>>();
        let cfg = config.clone();
        let handle = std::thread::spawn(move || worker_loop(cfg, rx, ready_tx));
        ready_rx
            .recv()
            .map_err(|_| AgentError::Other("worker thread died before ready".into()))??;
        Ok(Self {
            tx: Arc::new(tx),
            handle: Some(handle),
            tools_summary: None,
        })
    }

    pub fn with_tools(mut self, registry: &ToolRegistry) -> Self {
        if registry.list().is_empty() {
            return self;
        }
        let lines: Vec<String> = registry
            .list()
            .iter()
            .map(|t| {
                format!(
                    "- {}: {}\n  parameters: {}",
                    t.name(),
                    t.description(),
                    t.parameters_schema()
                )
            })
            .collect();
        self.tools_summary = Some(format!(
            "You have access to these tools:\n{}\n\n\
             To call tools, respond with ONLY JSON. Two accepted shapes:\n\
             - One call:     {{\"tool\": \"<name>\", \"args\": {{ ... }} }}\n\
             - Many calls:   [{{\"tool\": \"<name>\", \"args\": {{ ... }} }}, ...]\n\
             Otherwise reply to the user with plain text.",
            lines.join("\n"),
        ));
        self
    }

    /// Decode `prefix` once and pin its KV cells. Subsequent inference calls
    /// whose tokenized prompt starts with the same tokens skip re-decoding
    /// the prefix. Returns the prefix length in tokens.
    pub async fn prime(&self, prefix: String) -> Result<usize> {
        let (reply, rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(Request::Prime { prefix, reply })
            .map_err(|_| AgentError::Other("worker thread dead".into()))?;
        rx.await
            .map_err(|_| AgentError::Other("worker dropped reply".into()))?
    }

    /// Convenience: primes the canonical ChatML prefix for a session whose
    /// system message is `system`. The agent's tools summary (if any) is
    /// appended exactly as it would be in a normal turn, so context messages
    /// that start with `Message::System(system)` hit the fast path.
    pub async fn prime_for_system(&self, system: &str) -> Result<usize> {
        let prefix = self.system_prefix(system);
        self.prime(prefix).await
    }

    /// Build the canonical system-prefix string this agent would emit for
    /// the given system message. Exposed so callers can also pre-tokenize or
    /// log what's getting primed.
    pub fn system_prefix(&self, system: &str) -> String {
        let mut out = String::from("<|im_start|>system\n");
        out.push_str(system);
        if let Some(summary) = &self.tools_summary {
            out.push_str("\n\n");
            out.push_str(summary);
        }
        out.push_str("<|im_end|>\n");
        out
    }
}

impl Agent for LocalAgent {
    fn run<'a>(&'a self, ctx: Context) -> BoxFuture<'a, Result<Response>> {
        let prompt = format_chatml(&ctx, self.tools_summary.as_deref());
        let tx = Arc::clone(&self.tx);
        let has_tools = self.tools_summary.is_some();
        Box::pin(async move {
            let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
            tx.send(Request::Infer {
                prompt,
                reply: reply_tx,
            })
            .map_err(|_| AgentError::Other("worker thread dead".into()))?;
            let text = reply_rx
                .await
                .map_err(|_| AgentError::Other("worker dropped reply".into()))??;
            if has_tools {
                let calls = extract_tool_calls(&text);
                if !calls.is_empty() {
                    return Ok(Response::ToolCalls(calls));
                }
            }
            Ok(Response::Message(text))
        })
    }
}

// ---------- Worker thread ----------

fn worker_loop(
    config: LocalConfig,
    rx: mpsc::Receiver<Request>,
    ready_tx: mpsc::Sender<Result<()>>,
) {
    let backend = shared_backend();
    let model_params = LlamaModelParams::default().with_n_gpu_layers(config.n_gpu_layers);
    let model = match LlamaModel::load_from_file(backend, &config.model_path, &model_params) {
        Ok(m) => m,
        Err(e) => {
            let _ = ready_tx.send(Err(AgentError::Other(format!("load model: {e}"))));
            return;
        }
    };
    let flash_policy = if config.flash_attention {
        llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED
    } else {
        llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_DISABLED
    };
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(config.n_ctx))
        .with_n_seq_max(2)
        .with_flash_attention_policy(flash_policy);
    let mut ctx = match model.new_context(backend, ctx_params) {
        Ok(c) => c,
        Err(e) => {
            let _ = ready_tx.send(Err(AgentError::Other(format!("create context: {e}"))));
            return;
        }
    };
    let _ = ready_tx.send(Ok(()));

    let mut primed: Vec<LlamaToken> = Vec::new();

    while let Ok(req) = rx.recv() {
        match req {
            Request::Infer { prompt, reply } => {
                let result = run_inference(&model, &mut ctx, &primed, &prompt, &config);
                let _ = reply.send(result);
            }
            Request::Prime { prefix, reply } => {
                let result = prime_prefix(&model, &mut ctx, &prefix, config.n_ctx).map(|tokens| {
                    let n = tokens.len();
                    primed = tokens;
                    n
                });
                let _ = reply.send(result);
            }
            Request::Shutdown => break,
        }
    }
    // Explicit drop order matters for CUDA: the context (with its GPU
    // buffers) must release before the model. Rust drops locals in
    // reverse declaration order, so this is already correct — these
    // calls just make it impossible for someone to silently break it.
    drop(ctx);
    drop(model);
}

impl Drop for LocalAgent {
    fn drop(&mut self) {
        // Tell the worker to exit its loop *before* tearing down the channel;
        // it then drops `LlamaContext` and `LlamaModel` in the worker thread
        // (the only safe place for CUDA cleanup). We join to make sure that
        // completes before the process exits — otherwise the OS kills the
        // worker mid-cleanup and the CUDA runtime aborts with "CUDA error".
        let _ = self.tx.send(Request::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Decode `prefix` once under [`PRIME_SEQ_ID`] and return its tokens. The
/// cells written here stay alive until the next prime; inference calls
/// reach them by copying the sequence into [`INFER_SEQ_ID`].
fn prime_prefix(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    prefix: &str,
    n_ctx: u32,
) -> Result<Vec<LlamaToken>> {
    let tokens = model
        .str_to_token(prefix, AddBos::Never)
        .map_err(|e| AgentError::Other(format!("tokenize prefix: {e}")))?;
    if tokens.is_empty() {
        return Ok(tokens);
    }
    if (tokens.len() as u32) >= n_ctx {
        return Err(AgentError::Other(format!(
            "prefix tokens ({}) >= n_ctx ({n_ctx}) — cannot prime",
            tokens.len()
        )));
    }
    ctx.clear_kv_cache();
    let mut batch = LlamaBatch::new(n_ctx as usize, 1);
    for (i, token) in tokens.iter().enumerate() {
        let is_last = i == tokens.len() - 1;
        batch
            .add(*token, i as i32, &[PRIME_SEQ_ID], is_last)
            .map_err(|e| AgentError::Other(format!("batch add: {e}")))?;
    }
    ctx.decode(&mut batch)
        .map_err(|e| AgentError::Other(format!("prefix decode: {e}")))?;
    tracing::info!(prefix_tokens = tokens.len(), "primed prefix");
    Ok(tokens)
}

fn run_inference(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    primed: &[LlamaToken],
    prompt: &str,
    config: &LocalConfig,
) -> Result<String> {
    let tokens = model
        .str_to_token(prompt, AddBos::Never)
        .map_err(|e| AgentError::Other(format!("tokenize: {e}")))?;

    // Always reset the working sequence; PRIME_SEQ_ID stays untouched so the
    // primed prefix survives across calls.
    ctx.clear_kv_cache_seq(Some(INFER_SEQ_ID as u32), None, None)
        .map_err(|e| AgentError::Other(format!("clear infer seq: {e}")))?;

    let start_pos = if !primed.is_empty()
        && primed.len() < tokens.len()
        && tokens[..primed.len()] == *primed
    {
        // Fast path: share the primed cells into the working seq, decode only
        // the suffix.
        ctx.copy_kv_cache_seq(PRIME_SEQ_ID, INFER_SEQ_ID, None, None)
            .map_err(|e| AgentError::Other(format!("copy primed cells: {e}")))?;
        tracing::debug!(
            prefix_tokens = primed.len(),
            suffix_tokens = tokens.len() - primed.len(),
            "inference fast path"
        );
        primed.len()
    } else {
        tracing::debug!(prompt_tokens = tokens.len(), "inference full decode");
        0
    };

    let mut batch = LlamaBatch::new(config.n_ctx as usize, 1);
    let suffix = &tokens[start_pos..];
    for (i, token) in suffix.iter().enumerate() {
        let pos = start_pos + i;
        let is_last = i == suffix.len() - 1;
        batch
            .add(*token, pos as i32, &[INFER_SEQ_ID], is_last)
            .map_err(|e| AgentError::Other(format!("batch add: {e}")))?;
    }
    ctx.decode(&mut batch)
        .map_err(|e| AgentError::Other(format!("decode: {e}")))?;

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(config.temperature),
        LlamaSampler::dist(config.seed),
    ]);

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut output = String::new();
    let mut n_cur = tokens.len();
    for _ in 0..config.max_tokens {
        let token = sampler.sample(ctx, -1);
        if model.is_eog_token(token) {
            break;
        }
        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .map_err(|e| AgentError::Other(format!("piece: {e}")))?;
        output.push_str(&piece);

        batch.clear();
        batch
            .add(token, n_cur as i32, &[INFER_SEQ_ID], true)
            .map_err(|e| AgentError::Other(format!("batch add: {e}")))?;
        ctx.decode(&mut batch)
            .map_err(|e| AgentError::Other(format!("decode: {e}")))?;
        n_cur += 1;
    }
    Ok(output)
}

// ---------- ChatML formatting ----------

fn format_chatml(ctx: &Context, tools_summary: Option<&str>) -> String {
    let mut out = String::new();

    // Merge the tools summary into the (single) system message, or synthesize one.
    let mut wrote_system = false;
    for msg in &ctx.messages {
        if let Message::System { content } = msg {
            out.push_str("<|im_start|>system\n");
            out.push_str(content);
            if let Some(summary) = tools_summary {
                out.push_str("\n\n");
                out.push_str(summary);
            }
            out.push_str("<|im_end|>\n");
            wrote_system = true;
            break;
        }
    }
    if !wrote_system
        && let Some(summary) = tools_summary
    {
        out.push_str("<|im_start|>system\n");
        out.push_str(summary);
        out.push_str("<|im_end|>\n");
    }

    for msg in &ctx.messages {
        match msg {
            Message::System { .. } => {} // already emitted
            Message::User { content } => {
                out.push_str("<|im_start|>user\n");
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
            Message::Assistant { content } => {
                out.push_str("<|im_start|>assistant\n");
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
            Message::Tool { name, content } => {
                out.push_str("<|im_start|>user\n<tool_response name=\"");
                out.push_str(name);
                out.push_str("\">\n");
                out.push_str(content);
                out.push_str("\n</tool_response><|im_end|>\n");
            }
        }
    }

    out.push_str("<|im_start|>assistant\n");
    out
}

