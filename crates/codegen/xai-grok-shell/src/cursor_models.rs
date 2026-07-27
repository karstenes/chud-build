//! Cursor model catalog helpers.
//!
//! Bundled Cursor entries live in `default_models.json` with
//! `api_backend = "cursor_agent"`. When the user is logged into Cursor, this
//! module can also merge live ids from `api.cursor.com/v0/models` (Basic auth
//! with the Cursor API key) so newly published Cursor models appear without a
//! CLI upgrade.

use std::num::NonZeroU64;

use indexmap::IndexMap;
use xai_grok_config_types::LazinessDetectorPerModelConfig;
use xai_grok_sampler::cursor_agent::{FALLBACK_MODELS, fetch_available_models};
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
    let ids: Vec<String> = FALLBACK_MODELS.iter().map(|s| (*s).to_string()).collect();
    merge_cursor_model_ids(catalog, &ids);
}

/// Fetch live Cursor model ids (Basic `api_key:`) and merge them into `catalog`.
///
/// Falls back to [`FALLBACK_MODELS`] when the network call fails. No-op when
/// Cursor is not logged in or no API key is available for Basic auth.
pub async fn merge_live_cursor_catalog(catalog: &mut IndexMap<String, ModelEntry>) {
    if !cursor_auth::is_logged_in() {
        return;
    }
    let Some(api_key) = cursor_api_key_for_catalog() else {
        merge_cursor_fallback_catalog(catalog);
        return;
    };
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
    {
        Ok(c) => c,
        Err(_) => {
            merge_cursor_fallback_catalog(catalog);
            return;
        }
    };
    let ids = fetch_available_models(&client, &api_key).await;
    merge_cursor_model_ids(catalog, &ids);
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
                "opus-4.6".to_owned(),
            ],
        );
        assert_eq!(catalog.len(), 2);
        assert!(catalog.contains_key("opus-4.6"));
    }
}
