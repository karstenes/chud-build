//! Cursor model catalog helpers.
//!
//! Bundled Cursor entries live in `default_models.json` with
//! `api_backend = "cursor_agent"`. When the user is logged into Cursor, this
//! module merges live ids from AgentService `GetUsableModels` (Bearer OAuth
//! token) so only account-usable wire ids are offered. Falls back to
//! `api.cursor.com/v0/models` (API key) and then [`FALLBACK_MODELS`].

use std::collections::HashSet;
use std::num::NonZeroU64;
use std::time::Duration;

use indexmap::IndexMap;
use xai_grok_config_types::LazinessDetectorPerModelConfig;
use xai_grok_sampler::cursor_agent::{
    DEFAULT_CLIENT_VERSION, FALLBACK_MODELS, LEGACY_ALIAS_MODELS, fetch_available_models,
    fetch_usable_models,
};
use xai_grok_sampling_types::ApiBackend;

use crate::agent::config::{ModelEntry, ModelInfo, default_agent_type};
use crate::cursor_auth;

/// Build a catalog entry for a Cursor AgentService model id.
pub fn cursor_model_entry(model_id: &str) -> ModelEntry {
    let name = display_name(model_id);
    ModelEntry {
        info: ModelInfo {
            id: Some(model_id.to_owned()),
            model: model_id.to_owned(),
            base_url: cursor_auth::agent_base_url(),
            name: Some(name.clone()),
            description: Some(format!("{name} via Cursor AgentService/Run")),
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            api_backend: ApiBackend::CursorAgent,
            auth_scheme: Default::default(),
            extra_headers: IndexMap::new(),
            query_params: IndexMap::new(),
            env_http_headers: IndexMap::new(),
            context_window: NonZeroU64::new(200_000).expect("200000 is non-zero"),
            auto_compact_threshold_percent: Some(80),
            system_prompt_label: Some(name),
            use_concise: false,
            agent_type: "cursor".to_owned(),
            inference_idle_timeout_secs: None,
            max_retries: None,
            hidden: false,
            user_selectable: true,
            supported_in_api: false,
            reasoning_effort: None,
            supports_reasoning_effort: false,
            reasoning_efforts: Vec::new(),
            supports_backend_search: false,
            compactions_remaining: None,
            compaction_at_tokens: None,
            show_model_fingerprint: false,
            stream_tool_calls: None,
            laziness_detector: LazinessDetectorPerModelConfig::default(),
        },
        api_key: None,
        env_key: None,
        auth_provider: None,
        api_base_url: None,
    }
}

fn display_name(model_id: &str) -> String {
    model_id
        .split(['-', '_'])
        .filter(|s| !s.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(c) => format!("{}{}", c.to_uppercase(), chars.as_str()),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Insert missing Cursor models from `ids` into `catalog` (does not overwrite).
pub fn merge_cursor_model_ids(catalog: &mut IndexMap<String, ModelEntry>, ids: &[String]) {
    for id in ids {
        let id = id.trim();
        if id.is_empty() || id == "default" {
            continue;
        }
        if catalog.contains_key(id) {
            continue;
        }
        catalog.insert(id.to_owned(), cursor_model_entry(id));
    }
}

/// Merge the static Cursor fallback catalog when the user is logged in.
pub fn merge_cursor_fallback_catalog(catalog: &mut IndexMap<String, ModelEntry>) {
    if !cursor_auth::is_logged_in() {
        return;
    }
    // Drop short aliases that AgentService rejects with Connect not_found.
    for alias in LEGACY_ALIAS_MODELS {
        catalog.shift_remove(*alias);
    }
    let ids: Vec<String> = FALLBACK_MODELS.iter().map(|s| (*s).to_string()).collect();
    merge_cursor_model_ids(catalog, &ids);
}

/// Fetch live Cursor model ids and merge them into `catalog`.
///
/// Order of preference:
/// 1. `AgentService/GetUsableModels` with OAuth bearer (matches Run)
/// 2. `api.cursor.com/v0/models` with API key (Cloud Agents catalog)
/// 3. [`FALLBACK_MODELS`]
///
/// When GetUsableModels succeeds with a non-empty list, bundled/fallback
/// Cursor entries not in that list are pruned so the picker cannot offer
/// ids that fail Run with Connect `not_found`.
pub async fn merge_live_cursor_catalog(catalog: &mut IndexMap<String, ModelEntry>) {
    if !cursor_auth::is_logged_in() {
        return;
    }

    let client = match build_cursor_http_client() {
        Ok(c) => c,
        Err(error) => {
            tracing::warn!(%error, "Cursor model client build failed; using fallback catalog");
            merge_cursor_fallback_catalog(catalog);
            return;
        }
    };

    if let Some(token) = cursor_auth::access_token() {
        match fetch_usable_models_multi_host(&client, &token).await {
            Ok(ids) if !ids.is_empty() => {
                tracing::info!(
                    count = ids.len(),
                    "merged Cursor models from AgentService GetUsableModels"
                );
                prune_unavailable_bundled_cursor(catalog, &ids);
                merge_cursor_model_ids(catalog, &ids);
                return;
            }
            Ok(_) => {
                tracing::warn!(
                    "Cursor GetUsableModels returned empty; falling back to secondary catalogs"
                );
            }
            Err(error) => {
                tracing::warn!(
                    %error,
                    "Cursor GetUsableModels failed; falling back to secondary catalogs"
                );
            }
        }
    } else {
        tracing::warn!(
            "Cursor is logged in but no access token is available for GetUsableModels"
        );
    }

    if let Some(api_key) = cursor_api_key_for_catalog() {
        let ids = fetch_available_models(&client, &api_key).await;
        merge_cursor_model_ids(catalog, &ids);
    }

    merge_cursor_fallback_catalog(catalog);
}

/// Try agent host first (same as Run), then auth host (`api2`) used by some
/// Cursor CLI builds for unary RPCs.
async fn fetch_usable_models_multi_host(
    client: &reqwest::Client,
    token: &str,
) -> Result<Vec<String>, String> {
    let bases = [
        cursor_auth::agent_base_url(),
        cursor_auth::auth_base_url(),
    ];
    let mut last_err = None;
    for base in bases {
        match fetch_usable_models(client, token, &base, DEFAULT_CLIENT_VERSION).await {
            Ok(ids) => {
                tracing::info!(
                    base = %base,
                    count = ids.len(),
                    "Cursor GetUsableModels succeeded"
                );
                return Ok(ids);
            }
            Err(error) => {
                tracing::warn!(base = %base, %error, "Cursor GetUsableModels attempt failed");
                last_err = Some(error);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| "GetUsableModels: no hosts tried".into()))
}

fn build_cursor_http_client() -> Result<reqwest::Client, reqwest::Error> {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .connect_timeout(Duration::from_secs(10))
        .tcp_nodelay(true)
        .http2_keep_alive_interval(Duration::from_secs(15))
        .http2_keep_alive_timeout(Duration::from_secs(5))
        .http2_keep_alive_while_idle(true)
        .build()
}

fn prune_unavailable_bundled_cursor(catalog: &mut IndexMap<String, ModelEntry>, usable: &[String]) {
    let usable: HashSet<&str> = usable.iter().map(|s| s.as_str()).collect();
    let mut bundled: HashSet<&str> = FALLBACK_MODELS.iter().copied().collect();
    for alias in LEGACY_ALIAS_MODELS {
        bundled.insert(*alias);
    }
    catalog.retain(|id, entry| {
        if !(entry.info.api_backend.is_cursor_agent() || entry.info.agent_type == "cursor") {
            return true;
        }
        if bundled.contains(id.as_str()) || LEGACY_ALIAS_MODELS.contains(&id.as_str()) {
            return usable.contains(id.as_str());
        }
        true
    });
}

fn cursor_api_key_for_catalog() -> Option<String> {
    for name in ["GROK_CURSOR_API_KEY", "CURSOR_API_KEY"] {
        if let Ok(value) = std::env::var(name) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_owned());
            }
        }
    }
    cursor_auth::load_credentials()
        .ok()
        .flatten()
        .and_then(|c| c.api_key)
}

/// Ensure cursor models remain tagged correctly after remote prefetch overlays.
pub fn reinforce_cursor_entry(entry: &mut ModelEntry) {
    if entry.info.api_backend.is_cursor_agent() || entry.info.agent_type == "cursor" {
        entry.info.api_backend = ApiBackend::CursorAgent;
        if entry.info.agent_type == default_agent_type() {
            entry.info.agent_type = "cursor".to_owned();
        }
        entry.info.supported_in_api = false;
        if !cursor_auth::is_trusted_agent_base_url(&entry.info.base_url)
            || entry.info.base_url.trim().is_empty()
        {
            entry.info.base_url = cursor_auth::agent_base_url();
        }
        entry.api_base_url = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_model_entry_is_agent_backend() {
        let entry = cursor_model_entry("composer-2.5");
        assert!(entry.info.api_backend.is_cursor_agent());
        assert_eq!(entry.info.agent_type, "cursor");
        assert!(!entry.info.supported_in_api);
        assert!(cursor_auth::is_trusted_agent_base_url(&entry.info.base_url));
    }

    #[test]
    fn merge_skips_existing_and_default() {
        let mut catalog = IndexMap::new();
        catalog.insert("composer-2.5".to_owned(), cursor_model_entry("composer-2.5"));
        merge_cursor_model_ids(
            &mut catalog,
            &[
                "composer-2.5".to_owned(),
                "default".to_owned(),
                "claude-4.6-opus-high".to_owned(),
            ],
        );
        assert_eq!(catalog.len(), 2);
        assert!(catalog.contains_key("claude-4.6-opus-high"));
    }

    #[test]
    fn prune_removes_unusable_bundled_and_legacy_aliases() {
        let mut catalog = IndexMap::new();
        catalog.insert("composer-2.5".to_owned(), cursor_model_entry("composer-2.5"));
        catalog.insert(
            "gpt-5.4-medium".to_owned(),
            cursor_model_entry("gpt-5.4-medium"),
        );
        catalog.insert("sonnet-4.6".to_owned(), cursor_model_entry("sonnet-4.6"));
        catalog.insert("custom-cursor".to_owned(), cursor_model_entry("custom-cursor"));
        prune_unavailable_bundled_cursor(&mut catalog, &["composer-2.5".to_owned()]);
        assert!(catalog.contains_key("composer-2.5"));
        assert!(catalog.contains_key("custom-cursor"));
        assert!(!catalog.contains_key("gpt-5.4-medium"));
        assert!(!catalog.contains_key("sonnet-4.6"));
    }
}
