//! Text-only Cursor `AgentService/Run` streaming client for the grok sampler.
//!
//! Implements the paced, bidirectional Connect-over-HTTP/2 transport used by
//! the `cursor-agent` CLI against `agentn.global.api5.cursor.sh`. Tools are
//! deferred: this module streams assistant text deltas only.
//!
//! ## Attribution
//!
//! Adapted from MIT-licensed sources:
//! - `1jehuang/jcode` — `crates/jcode-provider-cursor-runtime/src/agent_transport.rs`
//!   (MIT, Jeremy Huang): Connect framing, paced `RunInput` frames, and
//!   `f1.f1.f1` answer extraction reverse-engineered from `cursor-agent`.
//! - `shunt` — `src/adapters/cursor/agent.rs` + `connect.rs` (MIT OR Apache-2.0):
//!   reqwest HTTP/2 body streaming, `ConnectFrameDecoder`, and gzip/end-frame
//!   handling.
//!
//! Wire contract (must match shunt/jcode):
//! - `POST {base}/agent.v1.AgentService/Run` over HTTP/2
//! - Connect framing: `[1 flag][4 BE len][payload]`; `FLAG_GZIP=0x01`,
//!   `FLAG_END=0x02`
//! - Request kept open while reading: frame0 `RunRequest`, frame1 env context,
//!   paced markers, then `f7:''` heartbeats every 5s

use std::collections::VecDeque;
use std::io::Read;
use std::time::{Duration, Instant};

use bytes::{Bytes, BytesMut};
use futures_util::{Stream, StreamExt};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::time::{Instant as TokioInstant, interval_at};
use xai_grok_sampling_types::{
    ConversationItem, ConversationRequest, ConversationResponse, ContentPart, SamplingError,
    StopReason,
};

use crate::events::{SamplingChannel, SamplingErrorInfo, SamplingEvent};
use crate::metrics::InferenceLatencyStats;
use crate::types::RequestId;

/// Default agent host for CLI/agent `AgentService/Run`.
pub const DEFAULT_AGENT_BASE_URL: &str = "https://agentn.global.api5.cursor.sh";

/// Client version advertised to Cursor's agent service. Must track a currently
/// served `cursor-agent` CLI build.
pub const DEFAULT_CLIENT_VERSION: &str = "cli-2026.07.08-0c04a8a";

/// Static fallback catalog when live discovery is unreachable.
///
/// IDs must match Cursor AgentService **wire** ids (after
/// [`normalize_agent_wire_model_id`]), not Cloud Agents / IDE slugs like
/// `cursor-grok-4.5-high`. Wrong ids fail Run with Connect `not_found`.
pub const FALLBACK_MODELS: &[&str] = &[
    "composer-2.5",
    "composer-2-fast",
    "composer-2",
    "grok-4.5-high",
    "grok-4.5-medium",
    "grok-4.5-xhigh",
    "grok-4.5-high-fast",
    "grok-4.5-fast-high",
    "grok-4.5-fast-medium",
    "gpt-5.4-high",
    "gpt-5.4-medium",
    "gpt-5.4-low",
    "claude-4.6-sonnet-medium",
    "claude-4.6-sonnet-medium-thinking",
    "claude-4.6-opus-high",
    "gemini-3.1-pro",
    "default",
];

/// Former short aliases that must never be advertised; pruned when live
/// discovery succeeds.
pub const LEGACY_ALIAS_MODELS: &[&str] =
    &["sonnet-4.6", "sonnet-4.6-thinking", "opus-4.6"];

/// Exact legacy CLI names that must keep a leading `cursor-` (or be the bare
/// word `cursor`). Everything else from GetUsableModels / Cloud Agents that
/// starts with `cursor-` is a display/catalog prefix and must be stripped
/// before `AgentService/Run` (e.g. `cursor-grok-4.5-high` → `grok-4.5-high`).
const CURSOR_PREFIX_LEGACY_WIRE_IDS: &[&str] = &[
    "cursor",
    "cursor-agent",
    "cursor-composer",
    "cursor-composer-fast",
    "cursor-plan",
    "cursor-ask",
];

/// Effort suffixes Cursor bakes into catalog / Cloud Agents compound ids.
/// Longest-first so `xhigh` wins over `high`. Variants like
/// `claude-opus-4-8-thinking` keep `thinking` in the base id; only the trailing
/// effort token is peeled into the `effort` param.
const CURSOR_EFFORT_SUFFIXES: &[&str] = &["xhigh", "medium", "high", "low", "max", "none"];

/// Resolved AgentService model selection (SDK/ACP style: base id + params).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CursorModelSelection {
    /// Base model id sent as ModelDetails.model_id (no effort / trailing-fast).
    pub model_id: String,
    /// Reasoning effort param (`effort`), when present on the catalog id.
    pub effort: Option<String>,
    /// `fast` ModelDetails param.
    pub fast: bool,
}

/// Normalize a catalog / picker id to the AgentService/Run wire id.
///
/// - `auto` → `default` (Cursor Auto)
/// - strip the `cursor-` prefix from GetUsableModels / Cloud Agents slugs
///   (`cursor-grok-4.5-high` → `grok-4.5-high`), except known legacy CLI names
pub fn normalize_agent_wire_model_id(model: &str) -> String {
    let trimmed = model.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if trimmed.eq_ignore_ascii_case("auto") {
        return "default".to_owned();
    }
    if CURSOR_PREFIX_LEGACY_WIRE_IDS
        .iter()
        .any(|legacy| trimmed.eq_ignore_ascii_case(legacy))
    {
        return trimmed.to_owned();
    }
    if let Some(rest) = trimmed.strip_prefix("cursor-") {
        let rest = rest.trim();
        if !rest.is_empty() {
            return rest.to_owned();
        }
    }
    trimmed.to_owned()
}

/// Split a catalog / compound Cursor id into base model + `effort` / `fast` params.
///
/// Cursor ACP/SDK select models as `grok-4.5[effort=high,fast=false]`. Cloud
/// Agents and some catalogs instead advertise compound slugs like
/// `cursor-grok-4.5-high-fast`. AgentService ModelDetails wants the **base**
/// id plus repeated kv params (`effort`, `fast`) — baking effort into the
/// name is what yields Connect `not_found` for reasoning models while bare
/// ids like `composer-2.5` / `gemini-3.1-pro` still work.
pub fn resolve_agent_model_selection(model: &str) -> CursorModelSelection {
    let mut id = normalize_agent_wire_model_id(model);
    if id.is_empty() {
        return CursorModelSelection {
            model_id: id,
            effort: None,
            fast: false,
        };
    }

    let mut fast = false;

    // Trailing `-fast` → SDK `fast=true` (composer-2.5-fast, grok-4.5-high-fast).
    if let Some(rest) = id.strip_suffix("-fast") {
        if !rest.is_empty() {
            fast = true;
            id = rest.to_owned();
        }
    }

    let mut effort = None;
    for suffix in CURSOR_EFFORT_SUFFIXES {
        let marker = format!("-{suffix}");
        if let Some(rest) = id.strip_suffix(marker.as_str()) {
            if rest.is_empty() {
                continue;
            }
            // `grok-4.5-fast-high` → base grok-4.5, effort high, fast true.
            let (base, nested_fast) = match rest.strip_suffix("-fast") {
                Some(base) if !base.is_empty() => (base.to_owned(), true),
                _ => (rest.to_owned(), false),
            };
            id = base;
            effort = Some((*suffix).to_owned());
            fast = fast || nested_fast;
            break;
        }
    }

    CursorModelSelection {
        model_id: id,
        effort,
        fast,
    }
}

const AGENT_PATH: &str = "/agent.v1.AgentService/Run";
const GET_USABLE_MODELS_PATH: &str = "/agent.v1.AgentService/GetUsableModels";
const MODELS_API_URL: &str = "https://api.cursor.com/v0/models";
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);
const FIRST_BYTE_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_PROMPT_CHARS: usize = 120_000;

const FLAG_GZIP: u8 = 0x01;
const FLAG_END: u8 = 0x02;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Stream Cursor agent turns as sampler events (text-only; tools deferred).
///
/// Takes owned inputs so the returned stream is `'static` and can be driven by
/// the sampler actor without tying lifetimes to [`crate::SamplingClient`].
pub fn stream_cursor_agent(
    client: reqwest::Client,
    token: String,
    base_url: String,
    client_version: String,
    model: String,
    prompt: String,
    cwd: String,
    request_id: RequestId,
    idle_timeout: Duration,
) -> impl Stream<Item = SamplingEvent> + Send + 'static {
    async_stream::stream! {
        let stream_start = Instant::now();
        let mut chunk_timestamps: Vec<Instant> = Vec::new();

        yield SamplingEvent::StreamStarted {
            request_id: request_id.clone(),
            timestamp_ms: chrono::Utc::now().timestamp_millis(),
        };

        // Catalog may use compound slugs (`grok-4.5-high-fast`); Run wants
        // base id + effort/fast ModelDetails params (SDK/ACP shape).
        let selection = resolve_agent_model_selection(&model);
        tracing::info!(
            target: crate::sampling_log::TARGET,
            event = "cursor_agent_model_selection",
            catalog_model = %model,
            wire_model = %selection.model_id,
            effort = selection.effort.as_deref().unwrap_or(""),
            fast = selection.fast,
            "resolved Cursor AgentService model selection"
        );
        let model = selection.model_id.clone();
        let frames = build_run_frames(&prompt, &selection, &cwd);
        let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(8);
        let (stop_tx, mut stop_rx) = oneshot::channel::<()>();
        let sender: JoinHandle<()> = tokio::spawn(async move {
            for (idx, frame) in frames.into_iter().enumerate() {
                if tx.send(Ok(frame)).await.is_err() {
                    return;
                }
                let pace = match idx {
                    0 => Duration::from_millis(1500),
                    1 => Duration::from_millis(800),
                    _ => Duration::from_millis(400),
                };
                tokio::select! {
                    _ = &mut stop_rx => return,
                    _ = tokio::time::sleep(pace) => {}
                }
            }
            let mut ticker = interval_at(
                TokioInstant::now() + HEARTBEAT_INTERVAL,
                HEARTBEAT_INTERVAL,
            );
            loop {
                tokio::select! {
                    _ = &mut stop_rx => return,
                    _ = ticker.tick() => {
                        if tx.send(Ok(heartbeat_frame())).await.is_err() {
                            return;
                        }
                    }
                }
            }
        });

        let body = reqwest::Body::wrap_stream(futures_util::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        }));

        let url = format!("{}{}", base_url.trim_end_matches('/'), AGENT_PATH);
        let wire_request_id = uuid::Uuid::new_v4().to_string();
        tracing::info!(
            target: crate::sampling_log::TARGET,
            event = "cursor_agent_post",
            url = %url,
            model = %model,
            client_version = %client_version,
            wire_request_id = %wire_request_id,
            "Cursor AgentService/Run POST"
        );
        let response = match client
            .post(&url)
            .bearer_auth(token)
            .header("connect-accept-encoding", "gzip,br")
            .header("connect-protocol-version", "1")
            .header("content-type", "application/connect+proto")
            .header("user-agent", "connect-es/1.6.1")
            .header("x-cursor-client-type", "cli")
            .header("x-cursor-client-version", client_version)
            .header("x-ghost-mode", "true")
            .header("x-request-id", &wire_request_id)
            .header("x-original-request-id", &wire_request_id)
            .body(body)
            .send()
            .await
        {
            Ok(response) => response,
            Err(error) => {
                let _ = stop_tx.send(());
                let _ = sender.await;
                tracing::warn!(
                    target: crate::sampling_log::TARGET,
                    event = "cursor_agent_http_error",
                    model = %model,
                    error = %error,
                    "Cursor AgentService/Run HTTP send failed"
                );
                let err = SamplingError::Http(error);
                yield SamplingEvent::Failed {
                    request_id: request_id.clone(),
                    error: SamplingErrorInfo::from(&err),
                };
                return;
            }
        };

        // Keep the paced request stream alive while we read the response.
        let _guard = TurnGuard {
            _stop: stop_tx,
            _sender: sender,
        };

        let status = response.status();
        if !status.is_success() {
            let message = response
                .text()
                .await
                .unwrap_or_else(|_| format!("HTTP {status}"));
            tracing::warn!(
                target: crate::sampling_log::TARGET,
                event = "cursor_agent_http_status",
                model = %model,
                status = status.as_u16(),
                message = %message.chars().take(500).collect::<String>(),
                "Cursor AgentService/Run non-success status"
            );
            let err = if status.as_u16() == 401 || status.as_u16() == 403 {
                SamplingError::Auth(message)
            } else {
                SamplingError::Api {
                    status,
                    message,
                    model_metadata: None,
                    retry_after_secs: None,
                    should_retry: None,
                }
            };
            yield SamplingEvent::Failed {
                request_id: request_id.clone(),
                error: SamplingErrorInfo::from(&err),
            };
            return;
        }

        let mut bytes = response.bytes_stream();
        let mut decoder = ConnectFrameDecoder::new();
        let mut pending: VecDeque<TurnEvent> = VecDeque::new();
        let mut got_text = false;
        let mut first_token_emitted = false;
        let mut content_acc = String::new();
        let mut chunk_index: u64 = 0;
        let mut message_chunk_count: u64 = 0;
        let mut finished = false;
        let mut failure: Option<SamplingError> = None;

        loop {
            if let Some(event) = pending.pop_front() {
                match event {
                    TurnEvent::Text(text) => {
                        got_text = true;
                        let now = Instant::now();
                        chunk_timestamps.push(now);
                        if !first_token_emitted {
                            first_token_emitted = true;
                            yield SamplingEvent::FirstToken {
                                request_id: request_id.clone(),
                            };
                        }
                        content_acc.push_str(&text);
                        message_chunk_count += 1;
                        yield SamplingEvent::ChannelToken {
                            request_id: request_id.clone(),
                            channel: SamplingChannel::Text,
                            text,
                            chunk_index,
                        };
                        chunk_index += 1;
                    }
                    TurnEvent::Failed(err) => {
                        failure = Some(err);
                        finished = true;
                    }
                    TurnEvent::End => {
                        finished = true;
                    }
                }
                if finished {
                    break;
                }
                continue;
            }
            if finished {
                break;
            }

            let budget = if got_text {
                idle_timeout
            } else {
                FIRST_BYTE_TIMEOUT
            };
            match tokio::time::timeout(budget, bytes.next()).await {
                Ok(Some(Ok(chunk))) => {
                    match decoder.push(&chunk) {
                        Ok(frames) => {
                            for frame in frames {
                                if let Some(event) = ingest_frame(&frame) {
                                    pending.push_back(event);
                                }
                            }
                        }
                        Err(error) => {
                            pending.push_back(TurnEvent::Failed(SamplingError::StreamError {
                                error_type: "frame".into(),
                                message: format!("cursor frame: {error}"),
                            }));
                        }
                    }
                }
                Ok(Some(Err(error))) => {
                    pending.push_back(TurnEvent::Failed(SamplingError::Http(error)));
                }
                Ok(None) => {
                    // Clean EOF. Leftover buffered bytes mean a truncated body.
                    match decoder.finish() {
                        Ok(()) => pending.push_back(TurnEvent::End),
                        Err(error) => {
                            pending.push_back(TurnEvent::Failed(SamplingError::StreamError {
                                error_type: "frame".into(),
                                message: format!("cursor frame: {error}"),
                            }));
                        }
                    }
                }
                Err(_) => {
                    if got_text {
                        // Server waits for tool exec we never send — end turn.
                        tracing::info!(
                            target: crate::sampling_log::TARGET,
                            event = "cursor_agent_idle_end",
                            model = %model,
                            content_len = content_acc.len(),
                            "Cursor agent idle after text; ending text-only turn"
                        );
                        pending.push_back(TurnEvent::End);
                    } else {
                        tracing::warn!(
                            target: crate::sampling_log::TARGET,
                            event = "cursor_agent_idle_timeout",
                            model = %model,
                            elapsed_secs = FIRST_BYTE_TIMEOUT.as_secs(),
                            "Cursor agent first-byte idle timeout"
                        );
                        pending.push_back(TurnEvent::Failed(SamplingError::IdleTimeout {
                            elapsed_secs: FIRST_BYTE_TIMEOUT.as_secs(),
                        }));
                    }
                }
            }
        }

        // Drain any remaining pending events that arrived with End/Failed.
        while let Some(event) = pending.pop_front() {
            match event {
                TurnEvent::Text(text) => {
                    let now = Instant::now();
                    chunk_timestamps.push(now);
                    if !first_token_emitted {
                        first_token_emitted = true;
                        yield SamplingEvent::FirstToken {
                            request_id: request_id.clone(),
                        };
                    }
                    content_acc.push_str(&text);
                    message_chunk_count += 1;
                    yield SamplingEvent::ChannelToken {
                        request_id: request_id.clone(),
                        channel: SamplingChannel::Text,
                        text,
                        chunk_index,
                    };
                    chunk_index += 1;
                }
                TurnEvent::Failed(err) => {
                    failure = Some(err);
                }
                TurnEvent::End => {}
            }
        }

        if let Some(err) = failure {
            let err = enrich_cursor_stream_error(err, &model);
            tracing::warn!(
                target: crate::sampling_log::TARGET,
                event = "cursor_agent_failed",
                model = %model,
                error = %err,
                content_len = content_acc.len(),
                "Cursor agent turn failed"
            );
            yield SamplingEvent::Failed {
                request_id: request_id.clone(),
                error: SamplingErrorInfo::from(&err),
            };
            return;
        }

        let stream_end = Instant::now();
        let metrics =
            InferenceLatencyStats::from_timestamps(stream_start, &chunk_timestamps, stream_end);
        tracing::info!(
            target: crate::sampling_log::TARGET,
            event = "cursor_agent_completed",
            model = %model,
            content_len = content_acc.len(),
            message_chunks = message_chunk_count,
            got_text,
            "Cursor agent turn completed"
        );
        let response = ConversationResponse {
            items: vec![ConversationItem::assistant_with_model(
                content_acc,
                model.to_string(),
            )],
            stop_reason: Some(StopReason::Stop),
            usage: None,
            cost_usd_ticks: None,
            message_chunks_emitted: message_chunk_count,
            doom_loop_signals: Vec::new(),
            stop_message: None,
        };
        yield SamplingEvent::Completed {
            request_id: request_id.clone(),
            response: Box::new(response),
            metrics,
        };
    }
}

/// Flatten a [`ConversationRequest`] into a CLI-style prompt string.
///
/// Mirrors jcode's `build_cli_prompt`: system preamble, then conversation
/// turns with `[tool_use …]` / `[tool_result …]` markers. Caps at
/// [`MAX_PROMPT_CHARS`] characters, keeping the tail.
pub fn build_prompt_from_conversation(request: &ConversationRequest) -> String {
    let mut system = String::new();
    for item in &request.items {
        if let ConversationItem::System(s) = item {
            if !system.is_empty() {
                system.push_str("\n\n");
            }
            system.push_str(s.content.as_ref());
        }
    }

    let mut out = String::new();
    if !system.trim().is_empty() {
        out.push_str("System:\n");
        out.push_str(system.trim());
        out.push_str("\n\n");
    }
    out.push_str("Conversation:\n");

    for item in &request.items {
        match item {
            ConversationItem::System(_) => {}
            ConversationItem::User(user) => {
                out.push_str("User:\n");
                for part in &user.content {
                    match part {
                        ContentPart::Text { text } => {
                            out.push_str(text.as_ref());
                            out.push('\n');
                        }
                        ContentPart::Image { .. } => {
                            out.push_str("[image]\n");
                        }
                    }
                }
                out.push('\n');
            }
            ConversationItem::Assistant(asst) => {
                out.push_str("Assistant:\n");
                if !asst.content.is_empty() {
                    out.push_str(asst.content.as_ref());
                    out.push('\n');
                }
                for call in &asst.tool_calls {
                    out.push_str("[tool_use ");
                    out.push_str(&call.name);
                    out.push_str(" input=");
                    out.push_str(call.arguments.as_ref());
                    out.push_str("]\n");
                }
                out.push('\n');
            }
            ConversationItem::ToolResult(result) => {
                out.push_str("User:\n");
                out.push_str("[tool_result ");
                out.push_str(&result.tool_call_id);
                out.push_str(" is_error=false]\n");
                out.push_str(result.content.as_ref());
                out.push('\n');
                for part in &result.images {
                    if matches!(part, ContentPart::Image { .. }) {
                        out.push_str("[image]\n");
                    }
                }
                out.push('\n');
            }
            ConversationItem::BackendToolCall(btc) => {
                out.push_str("Assistant:\n");
                out.push_str(&btc.text_summary());
                out.push_str("\n\n");
            }
            ConversationItem::Reasoning(_) => {}
        }
    }

    out.push_str("Assistant:\n");

    if out.chars().count() <= MAX_PROMPT_CHARS {
        return out;
    }

    let mut kept = out.chars().rev().take(MAX_PROMPT_CHARS).collect::<Vec<_>>();
    kept.reverse();
    let tail: String = kept.into_iter().collect();
    format!("[Earlier conversation truncated to fit prompt limits]\n\n{tail}")
}

/// Fetch live model ids from `api.cursor.com` (Basic auth with `api_key:`).
/// Falls back to [`FALLBACK_MODELS`] on error.
///
/// Prefer [`fetch_usable_models`] for AgentService/Run — that catalog is what
/// Run accepts. This Cloud Agents list can advertise ids the CLI wire rejects.
pub async fn fetch_available_models(client: &reqwest::Client, api_key: &str) -> Vec<String> {
    match fetch_available_models_inner(client, api_key).await {
        Ok(models) if !models.is_empty() => models,
        _ => FALLBACK_MODELS.iter().map(|s| (*s).to_string()).collect(),
    }
}

async fn fetch_available_models_inner(
    client: &reqwest::Client,
    api_key: &str,
) -> Result<Vec<String>, String> {
    let response = client
        .get(MODELS_API_URL)
        .basic_auth(api_key, Some(""))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !response.status().is_success() {
        return Err(format!("HTTP {}", response.status()));
    }
    #[derive(serde::Deserialize)]
    struct CursorModelsResponse {
        #[serde(default)]
        models: Vec<String>,
    }
    let parsed: CursorModelsResponse = response.json().await.map_err(|e| e.to_string())?;
    let mut seen = std::collections::HashSet::new();
    Ok(parsed
        .models
        .into_iter()
        .map(|m| normalize_agent_wire_model_id(&m))
        .filter(|m| !m.is_empty() && seen.insert(m.clone()))
        .collect())
}

/// Fetch account-usable AgentService model ids via unary
/// `AgentService/GetUsableModels` (Bearer access token, `application/proto`).
///
/// This is the authoritative catalog for [`stream_cursor_agent`]. Returns
/// `Err` on transport/HTTP failure and `Ok(vec![])` only when the response
/// decodes successfully with no models.
pub async fn fetch_usable_models(
    client: &reqwest::Client,
    access_token: &str,
    base_url: &str,
    client_version: &str,
) -> Result<Vec<String>, String> {
    let url = format!(
        "{}{}",
        base_url.trim_end_matches('/'),
        GET_USABLE_MODELS_PATH
    );
    let response = client
        .post(&url)
        .bearer_auth(access_token)
        .header("content-type", "application/proto")
        .header("te", "trailers")
        .header("x-cursor-client-type", "cli")
        .header("x-cursor-client-version", client_version)
        .header("x-ghost-mode", "true")
        // Empty GetUsableModelsRequest protobuf.
        .body(Vec::<u8>::new())
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !response.status().is_success() {
        return Err(format!("HTTP {}", response.status()));
    }
    let bytes = response.bytes().await.map_err(|e| e.to_string())?;
    decode_usable_models_response(&bytes)
}

fn decode_usable_models_response(payload: &[u8]) -> Result<Vec<String>, String> {
    let owned;
    let body: &[u8] = match connect_unary_payload(payload) {
        Ok(Some(bytes)) => {
            owned = bytes;
            &owned
        }
        Ok(None) => payload,
        Err(error) => return Err(error),
    };
    let mut models = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for field in iter_fields(body) {
        if field.field != 1 || field.wire != 2 {
            continue;
        }
        if let Some(id) = extract_model_details_id(field.data) {
            let id = normalize_agent_wire_model_id(&id);
            if id.is_empty() || !seen.insert(id.clone()) {
                continue;
            }
            models.push(id);
        }
    }
    Ok(models)
}

/// Prefer a Connect unary data frame when present; otherwise treat `payload`
/// as a raw protobuf body. Supports gzip-compressed Connect frames.
fn connect_unary_payload(payload: &[u8]) -> Result<Option<Vec<u8>>, String> {
    if payload.len() < 5 {
        return Ok(None);
    }
    let mut offset = 0;
    while offset + 5 <= payload.len() {
        let flags = payload[offset];
        let len = u32::from_be_bytes([
            payload[offset + 1],
            payload[offset + 2],
            payload[offset + 3],
            payload[offset + 4],
        ]) as usize;
        let frame_end = offset + 5 + len;
        if frame_end > payload.len() {
            return Ok(None);
        }
        if flags & FLAG_END == 0 {
            let frame = &payload[offset + 5..frame_end];
            if flags & FLAG_GZIP != 0 {
                return decode_gzip_frame(frame)
                    .map(Some)
                    .map_err(|e| format!("GetUsableModels gzip: {e}"));
            }
            return Ok(Some(frame.to_vec()));
        }
        offset = frame_end;
    }
    Ok(None)
}

fn extract_model_details_id(model_details: &[u8]) -> Option<String> {
    for field in iter_fields(model_details) {
        if field.field == 1 && field.wire == 2 {
            return std::str::from_utf8(field.data)
                .ok()
                .map(|s| s.to_string());
        }
    }
    None
}

fn enrich_cursor_stream_error(err: SamplingError, model: &str) -> SamplingError {
    match err {
        SamplingError::StreamError {
            error_type,
            message,
        } if error_type == "not_found" => SamplingError::StreamError {
            error_type,
            message: format!(
                "Cursor model '{model}' is not available for this account/session \
                 (Connect not_found). Reasoning models need base id + effort/fast \
                 params (e.g. grok-4.5 with effort=high), not compound Cloud Agents \
                 slugs. Prefer GetUsableModels ids or composer-2.5. Upstream: {message}"
            ),
        },
        other => other,
    }
}

// ---------------------------------------------------------------------------
// Turn internals
// ---------------------------------------------------------------------------

struct TurnGuard {
    _stop: oneshot::Sender<()>,
    _sender: JoinHandle<()>,
}

enum TurnEvent {
    Text(String),
    Failed(SamplingError),
    End,
}

fn ingest_frame(frame: &ConnectFrame) -> Option<TurnEvent> {
    if frame.flags & FLAG_END != 0 {
        if let Some(error) = parse_connect_error(&frame.payload) {
            return Some(TurnEvent::Failed(SamplingError::StreamError {
                error_type: error.code,
                message: error.message,
            }));
        }
        return Some(TurnEvent::End);
    }

    let decompressed;
    let payload = if frame.flags & FLAG_GZIP != 0 {
        match decode_gzip_frame(&frame.payload) {
            Ok(bytes) => {
                decompressed = bytes;
                &decompressed[..]
            }
            Err(error) => {
                return Some(TurnEvent::Failed(SamplingError::StreamError {
                    error_type: "gzip".into(),
                    message: format!("cursor gzip: {error}"),
                }));
            }
        }
    } else {
        &frame.payload[..]
    };

    extract_answer_text(payload).map(TurnEvent::Text)
}

// ---------------------------------------------------------------------------
// Connect framing (from shunt connect.rs)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
struct ConnectFrame {
    flags: u8,
    payload: Bytes,
}

fn encode_connect_frame(payload: impl AsRef<[u8]>, flags: u8) -> Bytes {
    let payload = payload.as_ref();
    let mut out = BytesMut::with_capacity(5 + payload.len());
    out.extend_from_slice(&[flags]);
    out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    out.extend_from_slice(payload);
    out.freeze()
}

#[derive(Default)]
struct ConnectFrameDecoder {
    buffer: BytesMut,
}

impl ConnectFrameDecoder {
    fn new() -> Self {
        Self::default()
    }

    fn push(&mut self, chunk: impl AsRef<[u8]>) -> Result<Vec<ConnectFrame>, ConnectError> {
        self.buffer.extend_from_slice(chunk.as_ref());
        self.drain(64 * 1024 * 1024)
    }

    fn drain(&mut self, max_payload: usize) -> Result<Vec<ConnectFrame>, ConnectError> {
        let mut out = Vec::new();
        loop {
            if self.buffer.len() < 5 {
                break;
            }
            let len = u32::from_be_bytes([
                self.buffer[1],
                self.buffer[2],
                self.buffer[3],
                self.buffer[4],
            ]) as usize;
            if len > max_payload {
                return Err(ConnectError::PayloadTooLarge {
                    length: len,
                    max: max_payload,
                });
            }
            if self.buffer.len() < 5 + len {
                break;
            }
            let mut raw = self.buffer.split_to(5 + len);
            out.push(ConnectFrame {
                flags: raw[0],
                payload: raw.split_off(5).freeze(),
            });
        }
        Ok(out)
    }

    fn finish(&self) -> Result<(), ConnectError> {
        if self.buffer.is_empty() {
            Ok(())
        } else {
            Err(ConnectError::TruncatedFrame {
                buffered: self.buffer.len(),
            })
        }
    }
}

const MAX_DECOMPRESSED_FRAME_BYTES: u64 = 64 * 1024 * 1024;

fn decode_gzip_frame(payload: &[u8]) -> Result<Vec<u8>, std::io::Error> {
    let decoder = flate2::read::GzDecoder::new(payload);
    let mut out = Vec::new();
    decoder
        .take(MAX_DECOMPRESSED_FRAME_BYTES + 1)
        .read_to_end(&mut out)?;
    if out.len() as u64 > MAX_DECOMPRESSED_FRAME_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "decompressed payload exceeds maximum allowed size",
        ));
    }
    Ok(out)
}

#[derive(serde::Deserialize)]
struct ConnectErrorPayload {
    error: ConnectErrorDetails,
}

#[derive(serde::Deserialize)]
struct ConnectErrorDetails {
    code: String,
    message: Option<String>,
}

struct ConnectEndError {
    code: String,
    message: String,
}

fn parse_connect_error(payload: &[u8]) -> Option<ConnectEndError> {
    if payload.is_empty() {
        return None;
    }
    let parsed: ConnectErrorPayload = serde_json::from_slice(payload).ok()?;
    Some(ConnectEndError {
        code: parsed.error.code,
        message: parsed
            .error
            .message
            .unwrap_or_else(|| "Connect error".to_string()),
    })
}

#[derive(Debug, Clone)]
enum ConnectError {
    PayloadTooLarge { length: usize, max: usize },
    TruncatedFrame { buffered: usize },
}

impl std::fmt::Display for ConnectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConnectError::PayloadTooLarge { length, max } => {
                write!(f, "Connect frame payload {length} exceeds max {max}")
            }
            ConnectError::TruncatedFrame { buffered } => {
                write!(
                    f,
                    "Cursor response truncated: {buffered} byte(s) of an incomplete frame remain"
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Protobuf codecs + RunRequest frames (text-only, from shunt/jcode)
// ---------------------------------------------------------------------------

fn encode_varint(mut value: u64, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push(((value as u8) & 0x7f) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn field_ld(field: u64, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 4);
    encode_varint((field << 3) | 2, &mut out);
    encode_varint(data.len() as u64, &mut out);
    out.extend_from_slice(data);
    out
}

fn field_varint(field: u64, value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    encode_varint(field << 3, &mut out);
    encode_varint(value, &mut out);
    out
}

fn field_str(field: u64, s: &str) -> Vec<u8> {
    field_ld(field, s.as_bytes())
}

fn connect_frame(payload: &[u8]) -> Bytes {
    encode_connect_frame(payload, 0)
}

fn encode_model_param(id: &str, value: &str) -> Vec<u8> {
    let mut kv = field_str(1, id);
    kv.extend(field_str(2, value));
    field_ld(3, &kv)
}

fn encode_model_meta(selection: &CursorModelSelection) -> Vec<u8> {
    let mut out = field_str(1, &selection.model_id);
    if let Some(effort) = selection.effort.as_deref() {
        out.extend(encode_model_param("effort", effort));
    }
    out.extend(encode_model_param(
        "fast",
        if selection.fast { "true" } else { "false" },
    ));
    out
}

/// Build Connect frames for a text-only agent turn (`mode=AGENT=1`, empty MCP tools).
fn build_run_frames(prompt: &str, selection: &CursorModelSelection, cwd: &str) -> Vec<Bytes> {
    let conv = uuid::Uuid::new_v4().to_string();
    let msg = uuid::Uuid::new_v4().to_string();
    let model_meta = encode_model_meta(selection);

    // frame 0: field 1 = RunRequest
    // messages: f2 { f1 { f1 { f1:prompt, f2:msg_id, f3:'', f4:1 } } }
    let mut inner = field_str(1, prompt);
    inner.extend(field_str(2, &msg));
    inner.extend(field_str(3, ""));
    inner.extend(field_varint(4, 1)); // AGENT mode
    let messages = field_ld(2, &field_ld(1, &field_ld(1, &inner)));

    let mut req = field_str(1, "");
    req.extend(messages);
    // f4 = empty mcp_tools (same bytes as field_str(4, ""))
    req.extend(field_str(4, ""));
    req.extend(field_str(5, &conv));
    req.extend(field_ld(9, &model_meta));
    req.extend(field_varint(12, 0));
    req.extend(field_ld(14, &field_str(1, "default")));
    req.extend(field_ld(14, &model_meta));
    req.extend(field_str(16, &conv));
    let frame0 = connect_frame(&field_ld(1, &req));

    // frame 1: field 2 = environment context
    let mut env = field_str(1, "linux");
    env.extend(field_str(2, cwd));
    env.extend(field_str(3, "bash"));
    env.extend(field_str(10, "UTC"));
    env.extend(field_str(11, cwd));
    env.extend(field_varint(14, 1));
    env.extend(field_varint(16, 1));
    env.extend(field_varint(19, 0));
    env.extend(field_varint(20, 0));
    env.extend(field_str(21, cwd));
    env.extend(field_varint(22, 0));
    let ctx = field_ld(
        2,
        &field_ld(10, &field_ld(1, &field_ld(1, &field_ld(4, &env)))),
    );
    let frame1 = connect_frame(&ctx);

    let mut frames = vec![frame0, frame1];
    frames.push(connect_frame(&field_ld(5, &field_str(1, ""))));
    frames.push(connect_frame(&field_ld(3, &field_str(3, ""))));
    for n in 1..=8u64 {
        let mut m = field_varint(1, n);
        m.extend(field_str(3, ""));
        frames.push(connect_frame(&field_ld(3, &m)));
    }
    frames
}

fn heartbeat_frame() -> Bytes {
    connect_frame(&field_ld(7, &[]))
}

// ---------------------------------------------------------------------------
// Response protobuf extraction
// ---------------------------------------------------------------------------

struct PbField<'a> {
    field: u64,
    wire: u8,
    data: &'a [u8],
}

fn iter_fields(mut buf: &[u8]) -> impl Iterator<Item = PbField<'_>> {
    std::iter::from_fn(move || {
        if buf.is_empty() {
            return None;
        }
        let (tag, rest) = read_varint(buf)?;
        let field = tag >> 3;
        let wire = (tag & 7) as u8;
        buf = rest;
        match wire {
            0 => {
                let (_v, rest) = read_varint(buf)?;
                buf = rest;
                Some(PbField {
                    field,
                    wire,
                    data: &[],
                })
            }
            2 => {
                let (len, rest) = read_varint(buf)?;
                let len = usize::try_from(len).ok()?;
                if rest.len() < len {
                    return None;
                }
                let data = &rest[..len];
                buf = &rest[len..];
                Some(PbField { field, wire, data })
            }
            5 => {
                if buf.len() < 4 {
                    return None;
                }
                buf = &buf[4..];
                Some(PbField {
                    field,
                    wire,
                    data: &[],
                })
            }
            1 => {
                if buf.len() < 8 {
                    return None;
                }
                buf = &buf[8..];
                Some(PbField {
                    field,
                    wire,
                    data: &[],
                })
            }
            _ => None,
        }
    })
}

fn read_varint(buf: &[u8]) -> Option<(u64, &[u8])> {
    let mut result = 0u64;
    let mut shift = 0u32;
    for (i, &byte) in buf.iter().enumerate() {
        result |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return Some((result, &buf[i + 1..]));
        }
        shift += 7;
        if shift >= 64 {
            return None;
        }
    }
    None
}

/// Assistant answer text delta: `f1.f1.f1` string.
fn extract_answer_text(payload: &[u8]) -> Option<String> {
    for f1 in iter_fields(payload) {
        if f1.field != 1 || f1.wire != 2 {
            continue;
        }
        for mid in iter_fields(f1.data) {
            if mid.field != 1 || mid.wire != 2 {
                continue;
            }
            for leaf in iter_fields(mid.data) {
                if leaf.field == 1
                    && leaf.wire == 2
                    && let Ok(text) = std::str::from_utf8(leaf.data)
                    && !text.is_empty()
                {
                    return Some(text.to_string());
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use xai_grok_sampling_types::ToolCall;

    #[test]
    fn encode_connect_frame_roundtrip() {
        let frame = encode_connect_frame(b"hello", 0);
        let mut decoder = ConnectFrameDecoder::new();
        let frames = decoder.push(&frame).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].flags, 0);
        assert_eq!(&frames[0].payload[..], b"hello");
        assert!(decoder.finish().is_ok());
    }

    #[test]
    fn build_run_frames_contains_model_name() {
        // PKCE-independent: frame building only needs prompt/model/cwd.
        let selection = resolve_agent_model_selection("composer-2.5");
        let frames = build_run_frames("PROMPT_MARKER", &selection, "/tmp");
        assert!(frames.len() >= 4);
        let hay = String::from_utf8_lossy(&frames[0]);
        assert!(hay.contains("PROMPT_MARKER"));
        assert!(hay.contains("composer-2.5"));
        assert!(hay.contains("fast"));
        for frame in &frames {
            assert!(frame.len() >= 5);
            let len = u32::from_be_bytes([frame[1], frame[2], frame[3], frame[4]]) as usize;
            assert_eq!(len + 5, frame.len());
            assert_eq!(frame[0], 0);
        }
    }

    #[test]
    fn build_run_frames_encodes_effort_param_not_suffix() {
        let selection = resolve_agent_model_selection("cursor-grok-4.5-high-fast");
        assert_eq!(selection.model_id, "grok-4.5");
        assert_eq!(selection.effort.as_deref(), Some("high"));
        assert!(selection.fast);
        let frames = build_run_frames("hi", &selection, "/tmp");
        let hay = String::from_utf8_lossy(&frames[0]);
        assert!(hay.contains("grok-4.5"));
        assert!(hay.contains("effort"));
        assert!(hay.contains("high"));
        // Must not send the compound catalog slug as the model id.
        assert!(!hay.contains("grok-4.5-high-fast"));
        assert!(!hay.contains("cursor-grok"));
    }

    #[test]
    fn extract_answer_text_reads_nested_chunk() {
        let leaf = field_str(1, "AUTH");
        let mid = field_ld(1, &leaf);
        let top = field_ld(1, &mid);
        assert_eq!(extract_answer_text(&top).as_deref(), Some("AUTH"));

        // Reasoning (f1.f4.f1) must not surface as answer text.
        let reasoning = field_ld(1, &field_ld(4, &field_str(1, "thinking")));
        assert_eq!(extract_answer_text(&reasoning), None);
    }

    #[test]
    fn build_prompt_from_conversation_truncation() {
        let mut request = ConversationRequest::default();
        request.items.push(ConversationItem::system("sys"));
        request.items.push(ConversationItem::user("hello"));
        request
            .items
            .push(ConversationItem::assistant("prior reply"));
        request.items.push(ConversationItem::Assistant(
            xai_grok_sampling_types::AssistantItem {
                content: Arc::<str>::from(""),
                tool_calls: vec![ToolCall {
                    id: Arc::<str>::from("call_1"),
                    name: "Read".into(),
                    arguments: Arc::<str>::from(r#"{"path":"/tmp/x"}"#),
                }],
                model_id: None,
                model_fingerprint: None,
                reasoning_effort: None,
            },
        ));
        request
            .items
            .push(ConversationItem::tool_result("call_1", "file contents"));

        let prompt = build_prompt_from_conversation(&request);
        assert!(prompt.contains("System:\nsys"));
        assert!(prompt.contains("User:\nhello"));
        assert!(prompt.contains("[tool_use Read input="));
        assert!(prompt.contains("[tool_result call_1 is_error=false]"));
        assert!(prompt.ends_with("Assistant:\n") || prompt.contains("Assistant:\n"));

        // Truncation keeps the tail within the cap.
        let mut huge = ConversationRequest::default();
        let filler = "x".repeat(MAX_PROMPT_CHARS + 5_000);
        huge.items.push(ConversationItem::user(filler.clone()));
        huge.items
            .push(ConversationItem::user("TAIL_MARKER_UNIQUE"));
        let truncated = build_prompt_from_conversation(&huge);
        assert!(truncated.starts_with("[Earlier conversation truncated to fit prompt limits]"));
        assert!(truncated.contains("TAIL_MARKER_UNIQUE"));
        assert!(truncated.chars().count() <= MAX_PROMPT_CHARS + 80);
    }

    #[test]
    fn heartbeat_is_stable() {
        assert_eq!(&heartbeat_frame()[..], &[0, 0, 0, 0, 2, 0x3a, 0x00]);
    }

    #[test]
    fn decode_usable_models_reads_model_id_field() {
        let mut details = field_str(1, "gpt-5.4-medium");
        details.extend(field_str(4, "GPT-5.4"));
        let response = field_ld(1, &details);
        let models = decode_usable_models_response(&response).unwrap();
        assert_eq!(models, vec!["gpt-5.4-medium".to_string()]);

        // Connect unary framing (flag 0 + BE length + payload).
        let framed = encode_connect_frame(&response, 0);
        let models = decode_usable_models_response(&framed).unwrap();
        assert_eq!(models, vec!["gpt-5.4-medium".to_string()]);
    }

    #[test]
    fn decode_usable_models_strips_cursor_prefix() {
        let details = field_str(1, "cursor-grok-4.5-high");
        let response = field_ld(1, &details);
        let models = decode_usable_models_response(&response).unwrap();
        assert_eq!(models, vec!["grok-4.5-high".to_string()]);
    }

    #[test]
    fn decode_usable_models_gzip_frame() {
        use flate2::Compression;
        use flate2::write::GzEncoder;
        use std::io::Write;

        let details = field_str(1, "cursor-grok-4.5-high");
        let response = field_ld(1, &details);
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&response).unwrap();
        let gz = encoder.finish().unwrap();
        let framed = encode_connect_frame(&gz, FLAG_GZIP);
        let models = decode_usable_models_response(&framed).unwrap();
        assert_eq!(models, vec!["grok-4.5-high".to_string()]);
    }

    #[test]
    fn normalize_wire_id_strips_cloud_agents_prefix() {
        assert_eq!(
            normalize_agent_wire_model_id("cursor-grok-4.5-high"),
            "grok-4.5-high"
        );
        assert_eq!(
            normalize_agent_wire_model_id("cursor-grok-4.5-high-fast"),
            "grok-4.5-high-fast"
        );
        assert_eq!(normalize_agent_wire_model_id("composer-2.5"), "composer-2.5");
        assert_eq!(normalize_agent_wire_model_id("auto"), "default");
        assert_eq!(normalize_agent_wire_model_id("cursor-agent"), "cursor-agent");
    }

    #[test]
    fn resolve_selection_splits_effort_and_fast() {
        assert_eq!(
            resolve_agent_model_selection("cursor-grok-4.5-high"),
            CursorModelSelection {
                model_id: "grok-4.5".into(),
                effort: Some("high".into()),
                fast: false,
            }
        );
        assert_eq!(
            resolve_agent_model_selection("grok-4.5-fast-high"),
            CursorModelSelection {
                model_id: "grok-4.5".into(),
                effort: Some("high".into()),
                fast: true,
            }
        );
        assert_eq!(
            resolve_agent_model_selection("composer-2.5-fast"),
            CursorModelSelection {
                model_id: "composer-2.5".into(),
                effort: None,
                fast: true,
            }
        );
        assert_eq!(
            resolve_agent_model_selection("claude-opus-4-8-thinking-low"),
            CursorModelSelection {
                model_id: "claude-opus-4-8-thinking".into(),
                effort: Some("low".into()),
                fast: false,
            }
        );
        assert_eq!(
            resolve_agent_model_selection("gemini-3.1-pro"),
            CursorModelSelection {
                model_id: "gemini-3.1-pro".into(),
                effort: None,
                fast: false,
            }
        );
    }

    #[test]
    fn enrich_not_found_mentions_model() {
        let err = enrich_cursor_stream_error(
            SamplingError::StreamError {
                error_type: "not_found".into(),
                message: "Error".into(),
            },
            "gpt-5.4-medium",
        );
        let msg = err.to_string();
        assert!(msg.contains("gpt-5.4-medium"));
        assert!(msg.contains("not_found"));
        assert!(!err.is_retryable());
    }
}
