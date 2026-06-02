//! Local-model [`Agent`] powered by llama.cpp via `llama-cpp-2`. A dedicated
//! OS thread owns the model/context; [`LocalAgent`] dispatches inference and
//! streaming requests to it.

use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::mpsc;

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::model::ChatTemplateResult;
use llama_cpp_2::model::LlamaChatTemplate;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::openai::OpenAIChatTemplateParams;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

use serde_json::Value;

use crate::agent::ToolCall;
use crate::message::{ContentPart, Message, Role};
use crate::tool::ToolRegistry;
use crate::tool_response::ToolResult;
use crate::{Agent, AgentError, BoxFuture, Context, Response, Result};

// ---------- Config ----------

#[derive(Clone)]
pub struct LocalConfig {
    pub model_path: String,
    pub n_ctx: u32,
    pub n_gpu_layers: u32,
    pub max_tokens: u32,
    pub temperature: f32,
    pub seed: u32,
    pub flash_attention: bool,
    /// Emit a "thinking" prefill marker for templates that gate CoT on it.
    pub enable_thinking: bool,
    /// Reasoning-format passed to llama.cpp's response parser (e.g. `"deepseek"`).
    pub reasoning_format: Option<String>,
    /// Render prompts via the GGUF's Jinja template. Set `false` to fall back
    /// to llama.cpp's legacy hardcoded format detector when the GGUF ships a
    /// broken Jinja template the strict parser rejects.
    pub use_jinja: bool,
    /// Run a dummy inference at startup to warm kernels and fail fast on
    /// broken chat templates.
    pub warmup: bool,
    /// When set, bypasses llama.cpp's `apply_chat_template_oaicompat` +
    /// streaming parser in favour of our own template rendering and tool-call
    /// parsing. For models whose embedded Jinja or upstream format detector
    /// produces unreliable tool calls (e.g. Llama 3.1).
    pub chat_template: Option<crate::templates::ChatTemplateKind>,
}

impl std::fmt::Debug for LocalConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalConfig")
            .field("model_path", &self.model_path)
            .field("n_ctx", &self.n_ctx)
            .field("n_gpu_layers", &self.n_gpu_layers)
            .field("max_tokens", &self.max_tokens)
            .field("temperature", &self.temperature)
            .field("seed", &self.seed)
            .field("flash_attention", &self.flash_attention)
            .field("enable_thinking", &self.enable_thinking)
            .field("reasoning_format", &self.reasoning_format)
            .field("use_jinja", &self.use_jinja)
            .field("warmup", &self.warmup)
            .field("chat_template", &self.chat_template)
            .finish()
    }
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
            enable_thinking: false,
            reasoning_format: None,
            use_jinja: true,
            warmup: true,
            chat_template: None,
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
    pub fn with_thinking(mut self, enabled: bool) -> Self {
        self.enable_thinking = enabled;
        self
    }
    pub fn with_reasoning_format(mut self, fmt: impl Into<String>) -> Self {
        self.reasoning_format = Some(fmt.into());
        self
    }
    pub fn with_warmup(mut self, enabled: bool) -> Self {
        self.warmup = enabled;
        self
    }
    pub fn with_chat_template(mut self, template: crate::templates::ChatTemplateKind) -> Self {
        self.chat_template = Some(template);
        self
    }
    pub fn with_jinja(mut self, enabled: bool) -> Self {
        self.use_jinja = enabled;
        self
    }
}

// ---------- KV cache sequence ids ----------

/// Two-sequence KV-cache design.
///
/// `WORK_SEQ_ID` runs the current inference: prefill writes here, sampling
/// extends it, cleared whole between calls. `CHECKPOINT_SEQ_ID` holds a
/// snapshot taken right after the previous call's prompt prefill (before
/// sampling); the next call reuses it via `copy_kv_cache_seq(CHECKPOINT →
/// WORK)` (cell-share, not a copy) then decodes only the new suffix.
///
/// Only full-clear and full-copy are used: mid-sequence truncation is
/// rejected by recurrent / SSM memory backends (e.g. Nemotron's Gated Delta
/// Net layers).
const WORK_SEQ_ID: i32 = 0;
const CHECKPOINT_SEQ_ID: i32 = 1;

// ---------- Process-global backend ----------

static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

fn shared_backend() -> &'static LlamaBackend {
    BACKEND.get_or_init(|| {
        llama_cpp_2::send_logs_to_tracing(
            llama_cpp_2::LogOptions::default()
                .with_logs_enabled(std::env::var("LLAMA_CPP_LOGS").is_ok()),
        );
        LlamaBackend::init().expect("llama backend init failed")
    })
}

// ---------- Worker requests ----------

enum Request {
    Infer {
        messages: Vec<Message>,
        messages_json: String,
        tools_json: Option<String>,
        tools_summary: Option<String>,
        json_schema: Option<String>,
        raw_grammar: Option<String>,
        parse_tool_calls: bool,
        reply: tokio::sync::oneshot::Sender<Result<String>>,
    },
    Stream {
        messages: Vec<Message>,
        messages_json: String,
        tools_json: Option<String>,
        tools_summary: Option<String>,
        json_schema: Option<String>,
        raw_grammar: Option<String>,
        sender: tokio::sync::mpsc::Sender<Result<crate::stream::Chunk>>,
    },
    Shutdown,
}

// ---------- Agent ----------

/// Agent driving a local GGUF model through llama.cpp's OpenAI-compat stack.
pub struct LocalAgent {
    tx: Arc<mpsc::Sender<Request>>,
    handle: Option<std::thread::JoinHandle<()>>,
    tools_json: Option<String>,
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
            tools_json: None,
            tools_summary: None,
        })
    }

    /// Register tools as the OpenAI tools-array JSON consumed by the worker.
    pub fn with_tools(mut self, registry: &ToolRegistry) -> Self {
        if registry.is_empty() {
            return self;
        }
        self.tools_json = Some(tools_to_openai_json(registry));
        self.tools_summary = crate::templates::build_tools_summary(registry);
        if let Some(t) = &self.tools_json {
            tracing::info!(bytes = t.len(), "tools_json built");
        }
        self
    }

    // No `prime_for_system`: the worker reuses the longest common prefix
    // between consecutive calls automatically (see `prefill_with_cache_reuse`).
}

impl Agent for LocalAgent {
    fn run<'a>(&'a self, ctx: Context) -> BoxFuture<'a, Result<Response>> {
        let tx = Arc::clone(&self.tx);
        let messages_json_result = messages_to_openai_json(&ctx.messages);
        let messages = ctx.messages.clone();
        let tools_json = self.tools_json.clone();
        let tools_summary = self.tools_summary.clone();
        let (json_schema, raw_grammar) = grammar_split(ctx.output_schema_request());
        // Tool-call parsing is mutually exclusive with structured output.
        let parse_tool_calls = ctx.output_schema_request().is_none() && tools_json.is_some();
        Box::pin(async move {
            let messages_json = messages_json_result?;
            let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
            tx.send(Request::Infer {
                messages,
                messages_json,
                tools_json,
                tools_summary,
                json_schema,
                raw_grammar,
                parse_tool_calls,
                reply: reply_tx,
            })
            .map_err(|_| AgentError::Other("worker thread dead".into()))?;
            let oai_msg_json = reply_rx
                .await
                .map_err(|_| AgentError::Other("worker dropped reply".into()))??;
            decode_openai_response(&oai_msg_json)
        })
    }
}

impl Drop for LocalAgent {
    fn drop(&mut self) {
        let _ = self.tx.send(Request::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
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
    // GGUFs without an embedded chat template are unsupported.
    let template = match model.chat_template(None) {
        Ok(t) => t,
        Err(e) => {
            let _ = ready_tx.send(Err(AgentError::Other(format!(
                "model has no embedded chat template: {e}"
            ))));
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
    // Fails fast on busted chat templates so the error surfaces at
    // `from_config`, not the user's first `agent.run`.
    if config.warmup
        && let Err(e) = run_warmup(&model, &template, &mut ctx, &config)
    {
        let _ = ready_tx.send(Err(e));
        return;
    }
    let _ = ready_tx.send(Ok(()));

    // Prompt tokens currently valid in WORK_SEQ; reused as LCP prefix
    // for the next call.
    let mut cached: Vec<LlamaToken> = Vec::new();

    while let Ok(req) = rx.recv() {
        match req {
            Request::Infer {
                messages,
                messages_json,
                tools_json,
                tools_summary,
                json_schema,
                raw_grammar,
                parse_tool_calls,
                reply,
            } => {
                let result = handle_infer(
                    &model,
                    &template,
                    &mut ctx,
                    &mut cached,
                    &config,
                    &messages,
                    &messages_json,
                    tools_json.as_deref(),
                    tools_summary.as_deref(),
                    json_schema.as_deref(),
                    raw_grammar.as_deref(),
                    parse_tool_calls,
                );
                let _ = reply.send(result);
            }
            Request::Stream {
                messages,
                messages_json,
                tools_json,
                tools_summary,
                json_schema,
                raw_grammar,
                sender,
            } => {
                let outcome = handle_stream(
                    &model,
                    &template,
                    &mut ctx,
                    &mut cached,
                    &config,
                    &messages,
                    &messages_json,
                    tools_json.as_deref(),
                    tools_summary.as_deref(),
                    json_schema.as_deref(),
                    raw_grammar.as_deref(),
                    &sender,
                );
                if let Err(e) = outcome {
                    let _ = sender.blocking_send(Err(e));
                }
                let _ = sender.blocking_send(Ok(crate::stream::Chunk::End));
            }
            Request::Shutdown => break,
        }
    }
    drop(ctx);
    drop(model);
}

/// Dummy inference through the full OAI-compat path to warm kernels and
/// surface Jinja errors at construction time.
fn run_warmup(
    model: &LlamaModel,
    template: &LlamaChatTemplate,
    ctx: &mut LlamaContext<'_>,
    config: &LocalConfig,
) -> Result<()> {
    let messages_json = "[\
        {\"role\":\"system\",\"content\":\"warmup\"},\
        {\"role\":\"user\",\"content\":\"ok\"}]";
    let params = build_oai_params(
        messages_json,
        None,
        None,
        None,
        config.reasoning_format.as_deref(),
        false,
        config.enable_thinking,
        /* add_generation_prompt = */ true,
        config.use_jinja,
    );
    let result = model
        .apply_chat_template_oaicompat(template, &params)
        .map_err(|e| AgentError::Other(format!("warmup apply_chat_template_oaicompat: {e}")))?;
    let tokens = model
        .str_to_token(&result.prompt, AddBos::Never)
        .map_err(|e| AgentError::Other(format!("warmup tokenize: {e}")))?;
    if tokens.is_empty() {
        return Ok(());
    }
    let mut batch = LlamaBatch::new(config.n_ctx as usize, 1);
    for (i, token) in tokens.iter().enumerate() {
        let is_last = i == tokens.len() - 1;
        batch
            .add(*token, i as i32, &[WORK_SEQ_ID], is_last)
            .map_err(|e| AgentError::Other(format!("warmup batch add: {e}")))?;
    }
    ctx.decode(&mut batch)
        .map_err(|e| AgentError::Other(format!("warmup decode: {e}")))?;
    // Exercise the sampler pipeline too; discard the result.
    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(config.temperature),
        LlamaSampler::dist(config.seed),
    ]);
    let _ = sampler.sample(ctx, -1);
    // Scoped clear: `clear_kv_cache()` can leave recurrent/SSM state in a
    // position-N limbo on some backends. Full-sequence clears are
    // universally supported.
    ctx.clear_kv_cache_seq(Some(WORK_SEQ_ID as u32), None, None)
        .map_err(|e| AgentError::Other(format!("warmup clear WORK seq: {e}")))?;
    tracing::info!(prompt_tokens = tokens.len(), "warmup complete");
    Ok(())
}

fn build_oai_params<'a>(
    messages_json: &'a str,
    tools_json: Option<&'a str>,
    json_schema: Option<&'a str>,
    grammar: Option<&'a str>,
    reasoning_format: Option<&'a str>,
    parse_tool_calls: bool,
    enable_thinking: bool,
    add_generation_prompt: bool,
    use_jinja: bool,
) -> OpenAIChatTemplateParams<'a> {
    OpenAIChatTemplateParams {
        messages_json,
        tools_json,
        tool_choice: None,
        json_schema,
        grammar,
        reasoning_format,
        chat_template_kwargs: None,
        add_generation_prompt,
        use_jinja,
        parallel_tool_calls: true,
        enable_thinking,
        add_bos: false,
        add_eos: false,
        parse_tool_calls,
    }
}

/// Split an [`OutputSchemaRequest`] into `(json_schema, raw_grammar)`: at
/// most one is `Some` — `Schema` routes through `params.json_schema` so
/// llama.cpp derives the GBNF; `Gbnf` is forwarded verbatim as `params.grammar`.
fn grammar_split(
    req: Option<&crate::structured::OutputSchemaRequest>,
) -> (Option<String>, Option<String>) {
    match req.map(|r| &r.grammar) {
        Some(crate::schema::GrammarSource::Schema(s)) => {
            (serde_json::to_string(&s.to_json_schema()).ok(), None)
        }
        Some(crate::schema::GrammarSource::Gbnf(g)) => (None, Some(g.clone())),
        None => (None, None),
    }
}

fn handle_infer(
    model: &LlamaModel,
    template: &LlamaChatTemplate,
    ctx: &mut LlamaContext<'_>,
    cached: &mut Vec<LlamaToken>,
    config: &LocalConfig,
    messages: &[Message],
    messages_json: &str,
    tools_json: Option<&str>,
    tools_summary: Option<&str>,
    json_schema: Option<&str>,
    raw_grammar: Option<&str>,
    parse_tool_calls: bool,
) -> Result<String> {
    // Schema turns stay on llama.cpp's path so the schema-derived GBNF
    // constrains output; our template path only handles tool-call turns.
    if parse_tool_calls
        && let Some(chat_template) = &config.chat_template
    {
        return handle_infer_with_template(
            model,
            ctx,
            cached,
            config,
            chat_template,
            messages,
            tools_summary,
            parse_tool_calls,
        );
    }
    let params = build_oai_params(
        messages_json,
        tools_json,
        json_schema,
        raw_grammar,
        config.reasoning_format.as_deref(),
        parse_tool_calls,
        config.enable_thinking,
        /* add_generation_prompt = */ true,
        config.use_jinja,
    );
    let result = model
        .apply_chat_template_oaicompat(template, &params)
        .map_err(|e| AgentError::Other(format!("apply_chat_template_oaicompat: {e}")))?;
    tracing::debug!(
        prompt_bytes = result.prompt.len(),
        grammar = result.grammar.is_some(),
        grammar_lazy = result.grammar_lazy,
        triggers = result.grammar_triggers.len(),
        stops = result.additional_stops.len(),
        "chat template rendered"
    );
    let text = generate(model, ctx, cached, &result, config)?;
    if parse_tool_calls {
        let parsed = result
            .parse_response_oaicompat(&text, /* is_partial = */ false)
            .map_err(|e| {
                AgentError::Other(format!("parse_response_oaicompat: {e}\nraw: {text}"))
            })?;
        Ok(parsed)
    } else {
        // Schema-only: grammar already constrained `text`. The OAI parser
        // chokes on plain JSON, so wrap it as an assistant envelope here.
        Ok(serde_json::json!({ "role": "assistant", "content": text }).to_string())
    }
}

/// Bypasses llama.cpp's chat template and OAI parser: renders the prompt via
/// [`crate::templates`], samples unconstrained, then scans the output with
/// [`crate::templates::parse_tool_calls`].
fn handle_infer_with_template(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    cached: &mut Vec<LlamaToken>,
    config: &LocalConfig,
    chat_template: &crate::templates::ChatTemplateKind,
    messages: &[Message],
    tools_summary: Option<&str>,
    parse_tool_calls: bool,
) -> Result<String> {
    use crate::templates::ChatTemplate;

    let prepared = prepare_messages_with_tools(messages, tools_summary);
    let prompt = chat_template.format(&prepared);
    let stops: Vec<String> = chat_template
        .stop_tokens()
        .iter()
        .map(|s| s.to_string())
        .collect();
    tracing::debug!(
        prompt_bytes = prompt.len(),
        stops = stops.len(),
        "our template rendered"
    );

    let text = generate_unconstrained(model, ctx, cached, config, &prompt, &stops)?;
    let cleaned = chat_template.clean_response(&text);

    if parse_tool_calls {
        let calls = crate::templates::parse_tool_calls(&cleaned);
        if !calls.is_empty() {
            let tool_calls: Vec<Value> = calls
                .into_iter()
                .map(|c| {
                    serde_json::json!({
                        "id": c.id,
                        "type": "function",
                        "function": {
                            "name": c.name,
                            "arguments": serde_json::to_string(&c.arguments)
                                .unwrap_or_else(|_| "{}".into()),
                        },
                    })
                })
                .collect();
            return Ok(serde_json::json!({
                "role": "assistant",
                "content": "",
                "tool_calls": tool_calls,
            })
            .to_string());
        }
    }
    Ok(serde_json::json!({ "role": "assistant", "content": cleaned }).to_string())
}

fn inject_schema_instruction(messages: &[Message], schema_json: &str) -> Vec<Message> {
    let instruction = format!(
        "Reply with a single JSON object matching this schema:\n{schema_json}\n\
         Output only the JSON object — no preamble, no markdown."
    );
    let mut prepared: Vec<Message> = messages
        .iter()
        .map(|m| {
            if m.role == Role::System {
                let text = m.as_text().unwrap_or("");
                Message::system(format!("{text}\n\n{instruction}"))
            } else {
                m.clone()
            }
        })
        .collect();
    if !prepared.iter().any(|m| m.role == Role::System) {
        prepared.insert(0, Message::system(instruction));
    }
    prepared
}

fn prepare_messages_with_tools(
    messages: &[Message],
    tools_summary: Option<&str>,
) -> Vec<Message> {
    let Some(summary) = tools_summary else {
        return messages.to_vec();
    };
    let mut prepared: Vec<Message> = messages
        .iter()
        .map(|m| {
            if m.role == Role::System {
                let text = m.as_text().unwrap_or("");
                Message::system(format!("{text}\n\n{summary}"))
            } else {
                m.clone()
            }
        })
        .collect();
    if !prepared.iter().any(|m| m.role == Role::System) {
        prepared.insert(0, Message::system(summary));
    }
    prepared
}

/// Streaming variant of [`handle_infer_with_template`]: emits `Chunk::Text`
/// per token, then runs the tool-call parser over the accumulated output and
/// emits `ToolEvent::Started`/`Arguments`/`Finished` for each call.
fn handle_stream_with_template(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    cached: &mut Vec<LlamaToken>,
    config: &LocalConfig,
    chat_template: &crate::templates::ChatTemplateKind,
    messages: &[Message],
    tools_summary: Option<&str>,
    parse_tool_calls: bool,
    json_schema: Option<&str>,
    raw_grammar: Option<&str>,
    sender: &tokio::sync::mpsc::Sender<Result<crate::stream::Chunk>>,
) -> Result<()> {
    use crate::stream::{Chunk, ToolEvent};
    use crate::templates::ChatTemplate;

    let mut prepared = prepare_messages_with_tools(messages, tools_summary);
    if let Some(schema) = json_schema {
        prepared = inject_schema_instruction(&prepared, schema);
    }
    let prompt = chat_template.format(&prepared);
    let stops: Vec<String> = chat_template
        .stop_tokens()
        .iter()
        .map(|s| s.to_string())
        .collect();

    let tokens = model
        .str_to_token(&prompt, AddBos::Never)
        .map_err(|e| AgentError::Other(format!("tokenize: {e}")))?;
    prefill_with_cache_reuse(ctx, cached, &tokens, config.n_ctx)?;
    let mut n_cur = tokens.len();

    let grammar_str = match (raw_grammar, json_schema) {
        (Some(g), _) => Some(g.to_string()),
        (None, Some(schema)) => Some(
            llama_cpp_2::json_schema_to_grammar(schema)
                .map_err(|e| AgentError::Other(format!("schema → grammar: {e}")))?,
        ),
        _ => None,
    };
    let mut sampler = match grammar_str.as_deref() {
        Some(g) => {
            let grammar_sampler = LlamaSampler::grammar(model, g, "root")
                .map_err(|e| AgentError::Other(format!("grammar sampler: {e}")))?;
            LlamaSampler::chain_simple([
                grammar_sampler,
                LlamaSampler::temp(config.temperature),
                LlamaSampler::dist(config.seed),
            ])
        }
        None => LlamaSampler::chain_simple([
            LlamaSampler::temp(config.temperature),
            LlamaSampler::dist(config.seed),
        ]),
    };
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut output = String::new();
    let mut batch = LlamaBatch::new(config.n_ctx as usize, 1);

    for _ in 0..config.max_tokens {
        let token = sampler
            .try_sample(ctx, -1)
            .map_err(|e| AgentError::Other(format!("sample: {e}")))?;
        if model.is_eog_token(token) {
            break;
        }
        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .map_err(|e| AgentError::Other(format!("piece: {e}")))?;
        output.push_str(&piece);

        if let Some(cut) = ends_with_any_stop(&output, &stops) {
            output.truncate(cut);
            break;
        }

        if sender.blocking_send(Ok(Chunk::Text(piece))).is_err() {
            return Ok(());
        }

        batch.clear();
        batch
            .add(token, n_cur as i32, &[WORK_SEQ_ID], true)
            .map_err(|e| AgentError::Other(format!("batch add: {e}")))?;
        ctx.decode(&mut batch)
            .map_err(|e| AgentError::Other(format!("decode: {e}")))?;
        n_cur += 1;
    }

    let cleaned = chat_template.clean_response(&output);
    if parse_tool_calls {
        for call in crate::templates::parse_tool_calls(&cleaned) {
            let id = call.id.clone();
            let _ = sender.blocking_send(Ok(Chunk::Tool(ToolEvent::Started {
                id: id.clone(),
                name: call.name,
            })));
            let args = serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".into());
            let _ = sender.blocking_send(Ok(Chunk::Tool(ToolEvent::Arguments {
                id: id.clone(),
                fragment: args,
            })));
            let _ = sender.blocking_send(Ok(Chunk::Tool(ToolEvent::Finished { id })));
        }
    }
    Ok(())
}

fn generate_unconstrained(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    cached_prompt: &mut Vec<LlamaToken>,
    config: &LocalConfig,
    prompt: &str,
    stops: &[String],
) -> Result<String> {
    let tokens = model
        .str_to_token(prompt, AddBos::Never)
        .map_err(|e| AgentError::Other(format!("tokenize: {e}")))?;
    let prompt_len = tokens.len();

    prefill_with_cache_reuse(ctx, cached_prompt, &tokens, config.n_ctx)?;

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(config.temperature),
        LlamaSampler::dist(config.seed),
    ]);
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut output = String::new();
    let mut batch = LlamaBatch::new(config.n_ctx as usize, 1);
    let mut n_cur = prompt_len;

    for _ in 0..config.max_tokens {
        let token = sampler
            .try_sample(ctx, -1)
            .map_err(|e| AgentError::Other(format!("sample: {e}")))?;
        if model.is_eog_token(token) {
            break;
        }
        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .map_err(|e| AgentError::Other(format!("piece: {e}")))?;
        output.push_str(&piece);

        if let Some(cut) = ends_with_any_stop(&output, stops) {
            output.truncate(cut);
            break;
        }

        batch.clear();
        batch
            .add(token, n_cur as i32, &[WORK_SEQ_ID], true)
            .map_err(|e| AgentError::Other(format!("batch add: {e}")))?;
        ctx.decode(&mut batch)
            .map_err(|e| AgentError::Other(format!("decode: {e}")))?;
        n_cur += 1;
    }
    Ok(output)
}

fn lcp(a: &[LlamaToken], b: &[LlamaToken]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Prefill the new prompt into `WORK_SEQ_ID`, reusing the previous prompt
/// via `CHECKPOINT_SEQ_ID`:
///
/// 1. Full-clear WORK.
/// 2. If the new prompt strictly extends the cached one: copy CHECKPOINT →
///    WORK (cell-share) and decode only the suffix; else decode the whole
///    prompt.
/// 3. Refresh CHECKPOINT from WORK so the next call can extend this boundary.
///
/// Only full-sequence clears and copies — recurrent / SSM backends reject
/// mid-sequence truncation.
fn prefill_with_cache_reuse(
    ctx: &mut LlamaContext<'_>,
    cached_prompt: &mut Vec<LlamaToken>,
    new_tokens: &[LlamaToken],
    n_ctx: u32,
) -> Result<()> {
    if new_tokens.is_empty() {
        return Err(AgentError::Other("empty prompt".into()));
    }
    if (new_tokens.len() as u32) >= n_ctx {
        return Err(AgentError::Other(format!(
            "prompt tokens ({}) >= n_ctx ({n_ctx})",
            new_tokens.len()
        )));
    }

    let common = lcp(cached_prompt, new_tokens);
    let can_reuse = !cached_prompt.is_empty()
        && common == cached_prompt.len()
        && common < new_tokens.len();

    ctx.clear_kv_cache_seq(Some(WORK_SEQ_ID as u32), None, None)
        .map_err(|e| AgentError::Other(format!("clear WORK seq: {e}")))?;

    let prefill_from = if can_reuse {
        ctx.copy_kv_cache_seq(CHECKPOINT_SEQ_ID, WORK_SEQ_ID, None, None)
            .map_err(|e| AgentError::Other(format!("copy CHECKPOINT → WORK: {e}")))?;
        tracing::debug!(
            cached = cached_prompt.len(),
            new = new_tokens.len(),
            reused = common,
            prefill = new_tokens.len() - common,
            "kv-cache reuse",
        );
        common
    } else {
        tracing::debug!(
            cached = cached_prompt.len(),
            new = new_tokens.len(),
            reused = 0,
            prefill = new_tokens.len(),
            "kv-cache full prefill"
        );
        0
    };

    decode_suffix_into_work(ctx, new_tokens, prefill_from, n_ctx)?;

    // Refresh CHECKPOINT before sampling so it always represents
    // "prompt prefilled, ready to generate" — never past sampled tokens
    // (which would diverge from a future template-rendered conversation).
    ctx.clear_kv_cache_seq(Some(CHECKPOINT_SEQ_ID as u32), None, None)
        .map_err(|e| AgentError::Other(format!("clear CHECKPOINT seq: {e}")))?;
    ctx.copy_kv_cache_seq(WORK_SEQ_ID, CHECKPOINT_SEQ_ID, None, None)
        .map_err(|e| AgentError::Other(format!("copy WORK → CHECKPOINT: {e}")))?;
    *cached_prompt = new_tokens.to_vec();
    Ok(())
}

fn decode_suffix_into_work(
    ctx: &mut LlamaContext<'_>,
    new_tokens: &[LlamaToken],
    from: usize,
    n_ctx: u32,
) -> Result<()> {
    let suffix = &new_tokens[from..];
    let mut batch = LlamaBatch::new(n_ctx as usize, 1);
    for (i, token) in suffix.iter().enumerate() {
        let pos = from + i;
        let is_last = i == suffix.len() - 1;
        batch
            .add(*token, pos as i32, &[WORK_SEQ_ID], is_last)
            .map_err(|e| AgentError::Other(format!("batch add: {e}")))?;
    }
    ctx.decode(&mut batch)
        .map_err(|e| AgentError::Other(format!("prefill decode: {e}")))?;
    Ok(())
}

fn generate(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    cached_prompt: &mut Vec<LlamaToken>,
    result: &ChatTemplateResult,
    config: &LocalConfig,
) -> Result<String> {
    let tokens = model
        .str_to_token(&result.prompt, AddBos::Never)
        .map_err(|e| AgentError::Other(format!("tokenize: {e}")))?;
    let prompt_len = tokens.len();

    prefill_with_cache_reuse(ctx, cached_prompt, &tokens, config.n_ctx)?;

    let mut sampler = build_sampler(model, result, config)?;
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut output = String::new();
    let mut batch = LlamaBatch::new(config.n_ctx as usize, 1);
    let mut n_cur = prompt_len;

    for _ in 0..config.max_tokens {
        let token = sampler
            .try_sample(ctx, -1)
            .map_err(|e| AgentError::Other(format!("sample: {e}")))?;
        if model.is_eog_token(token) {
            break;
        }
        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .map_err(|e| AgentError::Other(format!("piece: {e}")))?;
        output.push_str(&piece);

        if let Some(cut) = ends_with_any_stop(&output, &result.additional_stops) {
            output.truncate(cut);
            break;
        }

        batch.clear();
        batch
            .add(token, n_cur as i32, &[WORK_SEQ_ID], true)
            .map_err(|e| AgentError::Other(format!("batch add: {e}")))?;
        ctx.decode(&mut batch)
            .map_err(|e| AgentError::Other(format!("decode: {e}")))?;
        n_cur += 1;
    }

    tracing::info!(
        generated_tokens = n_cur - prompt_len,
        output_chars = output.len(),
        output = %output,
        "inference done"
    );
    Ok(output)
}

fn ends_with_any_stop(output: &str, stops: &[String]) -> Option<usize> {
    for stop in stops {
        if !stop.is_empty() && output.ends_with(stop) {
            return Some(output.len() - stop.len());
        }
    }
    None
}

fn build_sampler(
    model: &LlamaModel,
    result: &ChatTemplateResult,
    config: &LocalConfig,
) -> Result<LlamaSampler> {
    if let Some(grammar) = result.grammar.as_deref() {
        let grammar_sampler = if result.grammar_lazy {
            let patterns: Vec<String> = result
                .grammar_triggers
                .iter()
                .filter(|t| t.token.is_none())
                .map(|t| t.value.clone())
                .collect();
            let tokens: Vec<LlamaToken> = result
                .grammar_triggers
                .iter()
                .filter_map(|t| t.token)
                .collect();
            tracing::debug!(
                patterns = patterns.len(),
                tokens = tokens.len(),
                "lazy grammar"
            );
            LlamaSampler::grammar_lazy_patterns(model, grammar, "root", &patterns, &tokens)
                .map_err(|e| AgentError::Other(format!("lazy grammar sampler: {e}")))?
        } else {
            tracing::debug!("strict grammar");
            LlamaSampler::grammar(model, grammar, "root")
                .map_err(|e| AgentError::Other(format!("grammar sampler: {e}")))?
        };
        Ok(LlamaSampler::chain_simple([
            grammar_sampler,
            LlamaSampler::temp(config.temperature),
            LlamaSampler::dist(config.seed),
        ]))
    } else {
        Ok(LlamaSampler::chain_simple([
            LlamaSampler::temp(config.temperature),
            LlamaSampler::dist(config.seed),
        ]))
    }
}

// ---------- Our Message / Tool types → OpenAI JSON ----------

fn messages_to_openai_json(messages: &[Message]) -> Result<String> {
    let mut out: Vec<Value> = Vec::with_capacity(messages.len());
    for m in messages {
        let role_str = match &m.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
            Role::Custom(s) => s.as_str(),
        };
        let mut obj = serde_json::Map::new();
        obj.insert("role".into(), Value::String(role_str.into()));
        if matches!(m.role, Role::Tool) {
            for part in m.content.iter() {
                if let ContentPart::ToolResponse { response } = part {
                    obj.insert(
                        "tool_call_id".into(),
                        Value::String(response.call_id.clone()),
                    );
                    let content = match &response.result {
                        ToolResult::Json(v) => serde_json::to_string(v).unwrap_or_default(),
                        ToolResult::Text(t) => t.clone(),
                        ToolResult::Error(e) => format!("Error: {e}"),
                        ToolResult::Binary { content_type, .. } => {
                            format!("[binary: {content_type}]")
                        }
                    };
                    obj.insert("content".into(), Value::String(content));
                    break;
                }
            }
            if !obj.contains_key("content") {
                obj.insert("content".into(), Value::String(m.content.to_text()));
            }
        } else {
            obj.insert("content".into(), Value::String(m.content.to_text()));
        }
        out.push(Value::Object(obj));
    }
    serde_json::to_string(&out).map_err(Into::into)
}

fn tools_to_openai_json(registry: &ToolRegistry) -> String {
    let tools: Vec<Value> = registry
        .definitions()
        .iter()
        .map(|def| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": def.name,
                    "description": def.description,
                    "parameters": def.input.to_json_schema(),
                }
            })
        })
        .collect();
    serde_json::to_string(&tools).unwrap_or_else(|_| "[]".to_string())
}

// ---------- OpenAI message → Response ----------

fn decode_openai_response(oai_msg_json: &str) -> Result<Response> {
    let v: Value = serde_json::from_str(oai_msg_json).map_err(|e| {
        AgentError::Other(format!(
            "parse_response_oaicompat returned non-JSON: {e}\nraw: {oai_msg_json}"
        ))
    })?;
    if let Some(calls) = v.get("tool_calls").and_then(|v| v.as_array())
        && !calls.is_empty()
    {
        let mut out = Vec::with_capacity(calls.len());
        for (i, tc) in calls.iter().enumerate() {
            let name = tc
                .pointer("/function/name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            // OpenAI encodes arguments as a JSON string; decode back to Value.
            let arguments = match tc.pointer("/function/arguments") {
                Some(Value::String(s)) => serde_json::from_str::<Value>(s)
                    .unwrap_or_else(|_| Value::Object(Default::default())),
                Some(other) => other.clone(),
                None => Value::Object(Default::default()),
            };
            let id = tc
                .get("id")
                .and_then(|v| v.as_str())
                .map(String::from)
                .unwrap_or_else(|| i.to_string());
            out.push(ToolCall {
                id,
                name,
                arguments,
            });
        }
        return Ok(Response::ToolCalls(out));
    }
    let content = v
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    Ok(Response::Message(content))
}

// ---------- Streaming ----------

/// Generation loop that feeds each piece into llama.cpp's incremental OAI
/// parser and forwards the deltas as [`crate::stream::Chunk`]s onto `sender`.
fn handle_stream(
    model: &LlamaModel,
    template: &LlamaChatTemplate,
    ctx: &mut LlamaContext<'_>,
    cached: &mut Vec<LlamaToken>,
    config: &LocalConfig,
    messages: &[Message],
    messages_json: &str,
    tools_json: Option<&str>,
    tools_summary: Option<&str>,
    json_schema: Option<&str>,
    raw_grammar: Option<&str>,
    sender: &tokio::sync::mpsc::Sender<Result<crate::stream::Chunk>>,
) -> Result<()> {
    if let Some(chat_template) = &config.chat_template {
        let is_tool_turn = tools_json.is_some() && json_schema.is_none() && raw_grammar.is_none();
        let is_schema_turn = json_schema.is_some() || raw_grammar.is_some();
        if is_tool_turn || is_schema_turn {
            return handle_stream_with_template(
                model,
                ctx,
                cached,
                config,
                chat_template,
                messages,
                tools_summary,
                is_tool_turn,
                json_schema,
                raw_grammar,
                sender,
            );
        }
    }
    let params = build_oai_params(
        messages_json,
        tools_json,
        json_schema,
        raw_grammar,
        config.reasoning_format.as_deref(),
        tools_json.is_some(),
        config.enable_thinking,
        /* add_generation_prompt = */ true,
        config.use_jinja,
    );
    let result = model
        .apply_chat_template_oaicompat(template, &params)
        .map_err(|e| AgentError::Other(format!("apply_chat_template_oaicompat: {e}")))?;

    let mut state = result
        .streaming_state_oaicompat()
        .map_err(|e| AgentError::Other(format!("streaming_state_oaicompat: {e}")))?;

    let tokens = model
        .str_to_token(&result.prompt, AddBos::Never)
        .map_err(|e| AgentError::Other(format!("tokenize: {e}")))?;
    let prompt_len = tokens.len();

    prefill_with_cache_reuse(ctx, cached, &tokens, config.n_ctx)?;

    let mut sampler = build_sampler(model, &result, config)?;
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut batch = LlamaBatch::new(config.n_ctx as usize, 1);
    let mut n_cur = prompt_len;
    let mut emitted_first_chunk = false;
    let mut tool_id_by_index: std::collections::HashMap<u64, String> =
        std::collections::HashMap::new();
    let mut max_index_seen: Option<u64> = None;

    for _ in 0..config.max_tokens {
        let token = sampler
            .try_sample(ctx, -1)
            .map_err(|e| AgentError::Other(format!("sample: {e}")))?;
        if model.is_eog_token(token) {
            break;
        }
        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .map_err(|e| AgentError::Other(format!("piece: {e}")))?;

        let deltas = state
            .update(&piece, true)
            .map_err(|e| AgentError::Other(format!("streaming update: {e}")))?;
        for delta_json in deltas {
            for chunk in
                parse_oai_delta_to_chunks(&delta_json, &mut tool_id_by_index, &mut max_index_seen)
            {
                emitted_first_chunk = true;
                if sender.blocking_send(Ok(chunk)).is_err() {
                    return Ok(());
                }
            }
        }

        batch.clear();
        batch
            .add(token, n_cur as i32, &[WORK_SEQ_ID], true)
            .map_err(|e| AgentError::Other(format!("batch add: {e}")))?;
        ctx.decode(&mut batch)
            .map_err(|e| AgentError::Other(format!("decode: {e}")))?;
        n_cur += 1;
    }

    // Final flush so the parser can emit pending fragments and close open
    // tool calls.
    let final_deltas = state
        .update("", false)
        .map_err(|e| AgentError::Other(format!("streaming flush: {e}")))?;
    for delta_json in final_deltas {
        for chunk in
            parse_oai_delta_to_chunks(&delta_json, &mut tool_id_by_index, &mut max_index_seen)
        {
            emitted_first_chunk = true;
            if sender.blocking_send(Ok(chunk)).is_err() {
                return Ok(());
            }
        }
    }

    // Close the highest-index tool call: index-transition only fires when
    // a *new* index appears, so the last one never gets a Finished otherwise.
    if let Some(last) = max_index_seen
        && let Some(last_id) = tool_id_by_index.get(&last).cloned()
    {
        let _ = sender.blocking_send(Ok(crate::stream::Chunk::Tool(
            crate::stream::ToolEvent::Finished { id: last_id },
        )));
    }
    tracing::debug!(
        generated_tokens = n_cur - prompt_len,
        emitted_any = emitted_first_chunk,
        "stream done"
    );
    Ok(())
}

/// Translate one OAI streaming-delta JSON into 0+ [`Chunk`]s.
///
/// OpenAI sends the canonical `id` only on the first delta of a tool call;
/// subsequent deltas correlate by `index`. `tool_id_by_index` carries the
/// index→id map so every emitted chunk uses the same id. `max_index_seen`
/// detects index transitions so previous tool calls get a `Finished` as
/// soon as a higher index appears, not at end-of-stream.
fn parse_oai_delta_to_chunks(
    delta_json: &str,
    tool_id_by_index: &mut std::collections::HashMap<u64, String>,
    max_index_seen: &mut Option<u64>,
) -> Vec<crate::stream::Chunk> {
    use crate::stream::{Chunk, ToolEvent};
    let mut out = Vec::new();
    let Ok(v) = serde_json::from_str::<Value>(delta_json) else {
        return out;
    };
    if let Some(content) = v.get("content").and_then(|c| c.as_str())
        && !content.is_empty()
    {
        out.push(Chunk::Text(content.to_string()));
    }
    if let Some(tool_calls) = v.get("tool_calls").and_then(|t| t.as_array()) {
        for (i, tc) in tool_calls.iter().enumerate() {
            let index = tc
                .get("index")
                .and_then(|v| v.as_u64())
                .unwrap_or(i as u64);
            if let Some(prev) = *max_index_seen
                && index > prev
                && let Some(prev_id) = tool_id_by_index.get(&prev).cloned()
            {
                out.push(Chunk::Tool(ToolEvent::Finished { id: prev_id }));
            }
            *max_index_seen = Some(max_index_seen.map_or(index, |p| p.max(index)));
            if let Some(real_id) = tc.get("id").and_then(|v| v.as_str())
                && !real_id.is_empty()
            {
                tool_id_by_index.insert(index, real_id.to_string());
            }
            let id = tool_id_by_index
                .get(&index)
                .cloned()
                .unwrap_or_else(|| format!("call_{index}"));
            if let Some(name) = tc.pointer("/function/name").and_then(|v| v.as_str())
                && !name.is_empty()
            {
                out.push(Chunk::Tool(ToolEvent::Started {
                    id: id.clone(),
                    name: name.to_string(),
                }));
            }
            if let Some(args) = tc.pointer("/function/arguments").and_then(|v| v.as_str())
                && !args.is_empty()
            {
                out.push(Chunk::Tool(ToolEvent::Arguments {
                    id,
                    fragment: args.to_string(),
                }));
            }
        }
    }
    out
}

impl crate::stream::StreamingAgent for LocalAgent {
    type Stream = std::pin::Pin<
        Box<dyn tokio_stream::Stream<Item = Result<crate::stream::Chunk>> + Send>,
    >;

    fn run_stream(&self, ctx: Context) -> Self::Stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<crate::stream::Chunk>>(64);
        let messages_json = match messages_to_openai_json(&ctx.messages) {
            Ok(s) => s,
            Err(e) => {
                return Box::pin(tokio_stream::iter(std::iter::once(Err(e))));
            }
        };
        let messages = ctx.messages.clone();
        let tools_json = self.tools_json.clone();
        let tools_summary = self.tools_summary.clone();
        let (json_schema, raw_grammar) = grammar_split(ctx.output_schema_request());
        if self
            .tx
            .send(Request::Stream {
                messages,
                messages_json,
                tools_json,
                tools_summary,
                json_schema,
                raw_grammar,
                sender: tx,
            })
            .is_err()
        {
            return Box::pin(tokio_stream::iter(std::iter::once(Err(
                AgentError::Other("worker thread dead".into()),
            ))));
        }
        Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
    }
}
