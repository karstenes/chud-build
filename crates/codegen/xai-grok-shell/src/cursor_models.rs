//! Cursor model catalog helpers.
//!
//! Bundled Cursor entries live in `default_models.json` with
//! `api_backend = "cursor_agent"`. When the user is logged into Cursor, this
//! module merges live ids from AgentService `GetUsableModels` (Bearer OAuth
//! token) so only account-usable wire ids are offered. Falls back to a
//! **minimal** safe list (composer / default) — never the Cloud Agents
//! `api.cursor.com/v0/models` catalog, which advertises ids AgentService/Run
//! rejects with Connect `not_found`.
//!
//! Compound GetUsableModels ids (`grok-4.5-high`, `gpt-5.4-medium`) are
//! collapsed to a **base** picker entry with a `reasoning_efforts` menu so
//! `/effort` / `/reasoning` work the same as for xAI Responses models.

use std::collections::HashSet;
use std::num::NonZeroU64;
use std::str::FromStr;
use std::time::Duration;

use indexmap::IndexMap;
use xai_grok_config_types::LazinessDetectorPerModelConfig;
use xai_grok_sampler::cursor_agent::{
    DEFAULT_CLIENT_VERSION, LEGACY_ALIAS_MODELS, fetch_usable_models,
    normalize_agent_wire_model_id, resolve_agent_model_selection,
};
use xai_grok_sampling_types::{ApiBackend, ReasoningEffort, ReasoningEffortOption};

use crate::agent::config::{ModelEntry, ModelInfo, default_agent_type};
use crate::cursor_auth;

/// Build a catalog entry for a Cursor AgentService model id.
pub fn cursor_model_entry(model_id: &str) -> ModelEntry {
    cursor_model_entry_with_efforts(model_id, &[])
}

fn cursor_model_entry_with_efforts(model_id: &str, efforts: &[ReasoningEffort]) -> ModelEntry {
    let selection = resolve_agent_model_selection(model_id);
    let wire_model = if selection.fast {
        format!("{}-fast", selection.model_id)
    } else {
        selection.model_id.clone()
    };
    let name = display_name(&wire_model);
    let mut entry = ModelEntry {
        info: ModelInfo {
            id: Some(model_id.to_owned()),
            model: wire_model,
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
    };
    apply_cursor_reasoning_efforts(&mut entry, efforts);
    entry
}

fn apply_cursor_reasoning_efforts(entry: &mut ModelEntry, efforts: &[ReasoningEffort]) {
    if efforts.is_empty() {
        return;
    }
    let default_value = if efforts.contains(&ReasoningEffort::High) {
        ReasoningEffort::High
    } else {
        efforts[0]
    };
    entry.info.reasoning_efforts = efforts
        .iter()
        .copied()
        .map(|value| ReasoningEffortOption {
            id: value.as_str().to_owned(),
            value,
            label: effort_label(value),
            description: Some(effort_description(value)),
            default: value == default_value,
        })
        .collect();
    entry.info.reasoning_effort = None;
    entry.info.derive_reasoning_effort_fields();
}

fn effort_label(value: ReasoningEffort) -> String {
    match value {
        ReasoningEffort::None => "No Effort".into(),
        ReasoningEffort::Minimal => "Minimal Effort".into(),
        ReasoningEffort::Low => "Low Effort".into(),
        ReasoningEffort::Medium => "Medium Effort".into(),
        ReasoningEffort::High => "High Effort".into(),
        ReasoningEffort::Xhigh => "Extra High Effort".into(),
        ReasoningEffort::Max => "Max Effort".into(),
    }
}

fn effort_description(value: ReasoningEffort) -> String {
    match value {
        ReasoningEffort::None => "Disable extended reasoning".into(),
        ReasoningEffort::Minimal => "Minimal reasoning".into(),
        ReasoningEffort::Low => "Faster, lighter reasoning".into(),
        ReasoningEffort::Medium => "Balanced reasoning".into(),
        ReasoningEffort::High => "Heavy reasoning".into(),
        ReasoningEffort::Xhigh => "Extended reasoning".into(),
        ReasoningEffort::Max => "Maximum reasoning".into(),
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

fn parse_effort_token(token: &str) -> Option<ReasoningEffort> {
    ReasoningEffort::from_str(token).ok()
}

fn effort_sort_key(value: ReasoningEffort) -> u8 {
    match value {
        ReasoningEffort::None => 0,
        ReasoningEffort::Minimal => 1,
        ReasoningEffort::Low => 2,
        ReasoningEffort::Medium => 3,
        ReasoningEffort::High => 4,
        ReasoningEffort::Xhigh => 5,
        ReasoningEffort::Max => 6,
    }
}

fn dedup_sorted_efforts(efforts: impl IntoIterator<Item = ReasoningEffort>) -> Vec<ReasoningEffort> {
    let mut out: Vec<ReasoningEffort> = Vec::new();
    for effort in efforts {
        if !out.contains(&effort) {
            out.push(effort);
        }
    }
    out.sort_by_key(|e| effort_sort_key(*e));
    out
}

/// Picker key for a Cursor base model. Avoids colliding with an existing
/// non-Cursor entry that already owns the bare id (e.g. xAI `grok-4.5`).
fn cursor_picker_key(
    catalog: &IndexMap<String, ModelEntry>,
    base: &str,
    fast: bool,
) -> String {
    let candidate = if fast {
        format!("{base}-fast")
    } else {
        base.to_owned()
    };
    match catalog.get(&candidate) {
        Some(existing)
            if !(existing.info.api_backend.is_cursor_agent()
                || existing.info.agent_type == "cursor") =>
        {
            format!("cursor-{candidate}")
        }
        _ => candidate,
    }
}

fn wire_model_for(base: &str, fast: bool) -> String {
    if fast {
        format!("{base}-fast")
    } else {
        base.to_owned()
    }
}

/// Insert / update Cursor models from live or fallback ids.
///
/// Compound effort slugs (`grok-4.5-high`) collapse onto a base picker entry
/// with a `reasoning_efforts` menu so `/effort` and `/reasoning` work.
pub fn merge_cursor_model_ids(catalog: &mut IndexMap<String, ModelEntry>, ids: &[String]) {
    // (base, fast) → discovered effort tokens (empty vec = bare model).
    let mut groups: IndexMap<(String, bool), Vec<ReasoningEffort>> = IndexMap::new();
    for id in ids {
        let selection = resolve_agent_model_selection(id);
        if selection.model_id.is_empty() || selection.model_id == "default" {
            continue;
        }
        let entry = groups
            .entry((selection.model_id.clone(), selection.fast))
            .or_default();
        if let Some(effort) = selection
            .effort
            .as_deref()
            .and_then(parse_effort_token)
        {
            if !entry.contains(&effort) {
                entry.push(effort);
            }
        }
    }

    for ((base, fast), efforts) in groups {
        let key = cursor_picker_key(catalog, &base, fast);
        let wire = wire_model_for(&base, fast);
        let effort_list = dedup_sorted_efforts(efforts);
        if let Some(existing) = catalog.get_mut(&key) {
            if existing.info.api_backend.is_cursor_agent() || existing.info.agent_type == "cursor"
            {
                existing.info.model = wire;
                existing.info.id = Some(key.clone());
                if !effort_list.is_empty() {
                    let merged = dedup_sorted_efforts(
                        existing
                            .info
                            .reasoning_efforts
                            .iter()
                            .map(|o| o.value)
                            .chain(effort_list.iter().copied()),
                    );
                    apply_cursor_reasoning_efforts(existing, &merged);
                }
            }
            continue;
        }
        catalog.insert(
            key.clone(),
            cursor_model_entry_with_efforts(&key, &effort_list),
        );
        if let Some(entry) = catalog.get_mut(&key) {
            entry.info.model = wire;
            entry.info.id = Some(key);
        }
    }
}

/// Rewrite Cursor catalog keys that still use a Cloud Agents `cursor-`
/// prefix **as the wire model**. Picker aliases that already point at a
/// distinct base `info.model` (e.g. key `cursor-grok-4.5`, model `grok-4.5`)
/// are left alone so they do not collide with xAI `grok-4.5`.
pub fn normalize_cursor_catalog_keys(catalog: &mut IndexMap<String, ModelEntry>) {
    let remaps: Vec<(String, String)> = catalog
        .iter()
        .filter_map(|(key, entry)| {
            if !(entry.info.api_backend.is_cursor_agent() || entry.info.agent_type == "cursor") {
                return None;
            }
            // Only rewrite when the entry still uses the prefixed key as its wire id.
            if entry.info.model != *key {
                return None;
            }
            let wire = normalize_agent_wire_model_id(key);
            if wire.is_empty() || wire == *key {
                None
            } else {
                Some((key.clone(), wire))
            }
        })
        .collect();
    for (old, new) in remaps {
        if catalog.contains_key(&new) {
            catalog.shift_remove(&old);
            continue;
        }
        if let Some(mut entry) = catalog.shift_remove(&old) {
            entry.info.id = Some(new.clone());
            entry.info.model = new.clone();
            entry.info.name = Some(display_name(&new));
            entry.info.description = Some(format!(
                "{} via Cursor AgentService/Run",
                display_name(&new)
            ));
            entry.info.system_prompt_label = Some(display_name(&new));
            catalog.insert(new, entry);
        }
    }
}

/// Merge the static Cursor fallback catalog when the user is logged in.
///
/// Uses a **minimal** safe set only. Broader static Cursor catalogs include
/// named models many accounts cannot run on AgentService; offering them floods
/// the picker with Connect `not_found` failures.
pub fn merge_cursor_fallback_catalog(catalog: &mut IndexMap<String, ModelEntry>) {
    if !cursor_auth::is_logged_in() {
        return;
    }
    normalize_cursor_catalog_keys(catalog);
    for alias in LEGACY_ALIAS_MODELS {
        catalog.shift_remove(*alias);
    }
    prune_all_cursor_except(catalog, SAFE_FALLBACK_MODELS);
    let ids: Vec<String> = SAFE_FALLBACK_MODELS
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    merge_cursor_model_ids(catalog, &ids);
}

/// Models safe to offer when GetUsableModels is unreachable.
const SAFE_FALLBACK_MODELS: &[&str] = &["composer-2.5", "composer-2-fast", "default"];

/// Fetch live Cursor model ids and merge them into `catalog`.
pub async fn merge_live_cursor_catalog(catalog: &mut IndexMap<String, ModelEntry>) {
    if !cursor_auth::is_logged_in() {
        return;
    }

    normalize_cursor_catalog_keys(catalog);

    let client = match build_cursor_http_client() {
        Ok(c) => c,
        Err(error) => {
            tracing::warn!(%error, "Cursor model client build failed; using safe fallback catalog");
            merge_cursor_fallback_catalog(catalog);
            return;
        }
    };

    if let Some(token) = cursor_auth::access_token() {
        match fetch_usable_models_multi_host(&client, &token).await {
            Ok(ids) if !ids.is_empty() => {
                let preview = ids.iter().take(40).cloned().collect::<Vec<_>>().join(",");
                tracing::info!(
                    count = ids.len(),
                    models = %preview,
                    "merged Cursor models from AgentService GetUsableModels"
                );
                prune_unavailable_cursor(catalog, &ids);
                merge_cursor_model_ids(catalog, &ids);
                return;
            }
            Ok(_) => {
                tracing::warn!(
                    "Cursor GetUsableModels returned empty; using safe composer fallback"
                );
            }
            Err(error) => {
                tracing::warn!(
                    %error,
                    "Cursor GetUsableModels failed; using safe composer fallback"
                );
            }
        }
    } else {
        tracing::warn!(
            "Cursor is logged in but no access token is available for GetUsableModels"
        );
    }

    merge_cursor_fallback_catalog(catalog);
}

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

fn selection_identity(model: &str) -> (String, bool) {
    let selection = resolve_agent_model_selection(model);
    (selection.model_id, selection.fast)
}

/// Remove every Cursor AgentService catalog entry whose base/fast identity is
/// not represented in `usable` (after peeling effort suffixes).
fn prune_unavailable_cursor(catalog: &mut IndexMap<String, ModelEntry>, usable: &[String]) {
    let usable: HashSet<(String, bool)> = usable
        .iter()
        .map(|s| selection_identity(s))
        .filter(|(base, _)| !base.is_empty())
        .collect();
    for alias in LEGACY_ALIAS_MODELS {
        catalog.shift_remove(*alias);
    }
    catalog.retain(|id, entry| {
        if !(entry.info.api_backend.is_cursor_agent() || entry.info.agent_type == "cursor") {
            return true;
        }
        let identity = selection_identity(&entry.info.model);
        if usable.contains(&identity) {
            return true;
        }
        // Also accept picker keys that resolve to a usable identity.
        usable.contains(&selection_identity(id))
    });
}

fn prune_all_cursor_except(catalog: &mut IndexMap<String, ModelEntry>, keep: &[&str]) {
    let keep: HashSet<(String, bool)> = keep
        .iter()
        .map(|s| selection_identity(s))
        .filter(|(base, _)| !base.is_empty())
        .collect();
    for alias in LEGACY_ALIAS_MODELS {
        catalog.shift_remove(*alias);
    }
    catalog.retain(|id, entry| {
        if !(entry.info.api_backend.is_cursor_agent() || entry.info.agent_type == "cursor") {
            return true;
        }
        let identity = selection_identity(&entry.info.model);
        keep.contains(&identity) || keep.contains(&selection_identity(id))
    });
}

/// Ensure cursor models remain tagged correctly after remote prefetch overlays.
pub fn reinforce_cursor_entry(entry: &mut ModelEntry) {
    if entry.info.api_backend.is_cursor_agent() || entry.info.agent_type == "cursor" {
        entry.info.api_backend = ApiBackend::CursorAgent;
        if entry.info.agent_type == default_agent_type() {
            entry.info.agent_type = "cursor".to_owned();
        }
        entry.info.supported_in_api = false;
        // Do not strip intentional picker aliases; only normalize the wire model.
        let selection = resolve_agent_model_selection(&entry.info.model);
        let wire = if selection.fast {
            format!("{}-fast", selection.model_id)
        } else {
            selection.model_id.clone()
        };
        if !wire.is_empty() && wire != entry.info.model {
            entry.info.model = wire;
        }
        if !entry.info.reasoning_efforts.is_empty() {
            entry.info.derive_reasoning_effort_fields();
        }
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
    fn merge_collapses_effort_variants_onto_base() {
        let mut catalog = IndexMap::new();
        merge_cursor_model_ids(
            &mut catalog,
            &[
                "cursor-grok-4.5-high".to_owned(),
                "grok-4.5-medium".to_owned(),
                "grok-4.5-high-fast".to_owned(),
                "composer-2.5".to_owned(),
                "default".to_owned(),
            ],
        );
        assert!(catalog.contains_key("grok-4.5"));
        assert!(catalog.contains_key("grok-4.5-fast"));
        assert!(catalog.contains_key("composer-2.5"));
        assert!(!catalog.contains_key("grok-4.5-high"));
        let grok = catalog.get("grok-4.5").unwrap();
        assert_eq!(grok.info.model, "grok-4.5");
        assert!(grok.info.supports_reasoning_effort);
        let values: Vec<_> = grok
            .info
            .reasoning_efforts
            .iter()
            .map(|o| o.value)
            .collect();
        assert!(values.contains(&ReasoningEffort::High));
        assert!(values.contains(&ReasoningEffort::Medium));
        let fast = catalog.get("grok-4.5-fast").unwrap();
        assert_eq!(fast.info.model, "grok-4.5-fast");
        assert!(fast.info.supports_reasoning_effort);
    }

    #[test]
    fn merge_uses_prefixed_key_when_xai_owns_base() {
        let mut catalog = IndexMap::new();
        let mut xai = cursor_model_entry("ignored");
        xai.info.api_backend = ApiBackend::Responses;
        xai.info.agent_type = "grok-build".to_owned();
        xai.info.model = "grok-4.5".to_owned();
        catalog.insert("grok-4.5".to_owned(), xai);

        merge_cursor_model_ids(&mut catalog, &["grok-4.5-high".to_owned()]);
        assert!(catalog.contains_key("cursor-grok-4.5"));
        let cursor = catalog.get("cursor-grok-4.5").unwrap();
        assert_eq!(cursor.info.model, "grok-4.5");
        assert!(cursor.info.supports_reasoning_effort);
        assert!(matches!(
            catalog.get("grok-4.5").unwrap().info.api_backend,
            ApiBackend::Responses
        ));
    }

    #[test]
    fn normalize_catalog_skips_alias_keys_with_distinct_wire_model() {
        let mut catalog = IndexMap::new();
        let mut entry = cursor_model_entry("cursor-grok-4.5");
        entry.info.model = "grok-4.5".to_owned();
        catalog.insert("cursor-grok-4.5".to_owned(), entry);
        normalize_cursor_catalog_keys(&mut catalog);
        assert!(catalog.contains_key("cursor-grok-4.5"));
        assert_eq!(catalog.get("cursor-grok-4.5").unwrap().info.model, "grok-4.5");
    }

    #[test]
    fn prune_removes_all_unusable_cursor_entries() {
        let mut catalog = IndexMap::new();
        catalog.insert("composer-2.5".to_owned(), cursor_model_entry("composer-2.5"));
        catalog.insert(
            "gpt-5.4-medium".to_owned(),
            cursor_model_entry("gpt-5.4-medium"),
        );
        catalog.insert(
            "gpt-5.6-luna-high".to_owned(),
            cursor_model_entry("gpt-5.6-luna-high"),
        );
        let mut xai = cursor_model_entry("ignored");
        xai.info.api_backend = ApiBackend::Responses;
        xai.info.agent_type = "grok-build".to_owned();
        catalog.insert("grok-4.5".to_owned(), xai);

        prune_unavailable_cursor(&mut catalog, &["composer-2.5".to_owned()]);
        assert!(catalog.contains_key("composer-2.5"));
        assert!(catalog.contains_key("grok-4.5"));
        assert!(!catalog.contains_key("gpt-5.4-medium"));
        assert!(!catalog.contains_key("gpt-5.6-luna-high"));
    }

    #[test]
    fn prune_keeps_base_when_usable_lists_effort_variant() {
        let mut catalog = IndexMap::new();
        merge_cursor_model_ids(
            &mut catalog,
            &["grok-4.5-high".to_owned(), "grok-4.5-medium".to_owned()],
        );
        prune_unavailable_cursor(&mut catalog, &["cursor-grok-4.5-high".to_owned()]);
        assert!(catalog.contains_key("grok-4.5"));
    }

    #[test]
    fn safe_fallback_keeps_only_composer() {
        let mut catalog = IndexMap::new();
        catalog.insert(
            "gpt-5.6-luna-high".to_owned(),
            cursor_model_entry("gpt-5.6-luna-high"),
        );
        catalog.insert("composer-2.5".to_owned(), cursor_model_entry("composer-2.5"));
        prune_all_cursor_except(&mut catalog, SAFE_FALLBACK_MODELS);
        assert!(catalog.contains_key("composer-2.5"));
        assert!(!catalog.contains_key("gpt-5.6-luna-high"));
    }
}
