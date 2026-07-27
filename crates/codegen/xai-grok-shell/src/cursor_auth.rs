//! Isolated Cursor OAuth account support for this grok-build fork.
//!
//! This intentionally does not reuse Grok's primary [`crate::auth::AuthManager`].
//! Cursor login, refresh, logout, and credential storage must never mutate xAI's
//! `auth.json`, call xAI logout/billing, or change the ACP primary-auth state.
//!
//! Credentials live only in `cursor-auth.json` under [`xai_grok_config::grok_home`].
//! The login flow mirrors Cursor's `loginDeepControl` + `/auth/poll` PKCE protocol
//! (same contract as the shunt reference implementation).

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use base64::Engine as _;
use chrono::{DateTime, Utc};
use rand::RngCore as _;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest as _, Sha256};

pub const CURSOR_AUTH_FILE_NAME: &str = "cursor-auth.json";
pub const CURSOR_AUTH_BASE_URL: &str = "https://api2.cursor.sh";
pub const CURSOR_AGENT_BASE_URL: &str = "https://agentn.global.api5.cursor.sh";
pub const CURSOR_LOGIN_URL: &str = "https://cursor.com/loginDeepControl";
pub const CURSOR_CLIENT_VERSION: &str = "cli-2026.07.08-0c04a8a";
pub const CURSOR_AUTH_REQUIRED_ERROR_KIND: &str = "cursor_auth_required";
pub const CURSOR_AUTH_REQUIRED_MESSAGE: &str =
    "Cursor authentication required; run `grok login --cursor` or set CURSOR_API_KEY";

const CURSOR_AUTH_BASE_URL_ENV: &str = "GROK_CURSOR_AUTH_BASE_URL";
const CURSOR_AGENT_BASE_URL_ENV: &str = "GROK_CURSOR_AGENT_BASE_URL";
const AUTH_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const REFRESH_WINDOW_SECS: i64 = 5 * 60;
const POLL_MAX_ATTEMPTS: u32 = 150;

/// On-disk Cursor credential schema (`camelCase`), stored separately at
/// `~/.grok/cursor-auth.json` with owner-only permissions.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
struct StoredCursorAuth {
    access_token: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    refresh_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    api_key: Option<String>,
}

/// Live Cursor credentials used by the sampler and session plumbing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CursorCredentials {
    pub access_token: String,
    pub api_key: Option<String>,
    pub email: Option<String>,
    pub user_id: Option<String>,
}

/// Account-scoped identity captured after a successful Cursor login.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CursorAccountSummary {
    pub email: Option<String>,
    pub user_id: Option<String>,
}

impl From<&CursorCredentials> for CursorAccountSummary {
    fn from(value: &CursorCredentials) -> Self {
        Self {
            email: value.email.clone(),
            user_id: value.user_id.clone(),
        }
    }
}

/// Per-request sync resolver used by the sampler. Refresh happens in the
/// isolated proactive loop; this read observes credential rotation performed
/// by this or another process writing `cursor-auth.json`.
#[derive(Clone, Debug, Default)]
pub struct CursorBearerResolver {
    seed_token: Option<String>,
}

impl CursorBearerResolver {
    pub(crate) fn from_credentials(creds: &CursorCredentials) -> Self {
        Self::from_token(creds.access_token.clone())
    }

    pub(crate) fn from_token(access_token: String) -> Self {
        Self {
            seed_token: Some(access_token),
        }
    }
}

impl xai_grok_sampler::BearerResolver for CursorBearerResolver {
    fn current_bearer(&self) -> Option<String> {
        if let Some(token) = process_auth_token_override() {
            return Some(token);
        }
        if let Ok(Some(creds)) = load_credentials() {
            return Some(creds.access_token);
        }
        self.seed_token.clone()
    }
}

/// Canonical Cursor login/refresh endpoint, with an explicit process-level
/// override for development proxies and hermetic tests.
pub fn auth_base_url() -> String {
    std::env::var(CURSOR_AUTH_BASE_URL_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(|value| value.trim().to_owned())
        .unwrap_or_else(|| CURSOR_AUTH_BASE_URL.to_owned())
}

/// Canonical Cursor agent inference endpoint, with an explicit process-level
/// override for development proxies and hermetic tests.
pub fn agent_base_url() -> String {
    std::env::var(CURSOR_AGENT_BASE_URL_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(|value| value.trim().to_owned())
        .unwrap_or_else(|| CURSOR_AGENT_BASE_URL.to_owned())
}

/// Process-level trust gate for agent base URLs. Empty (caller default) and the
/// currently configured agent base are trusted; anything else is rejected.
pub fn is_trusted_agent_base_url(url: &str) -> bool {
    let candidate = url.trim().trim_end_matches('/');
    candidate.is_empty() || candidate == agent_base_url().trim().trim_end_matches('/')
}

pub fn auth_file_path() -> PathBuf {
    xai_grok_config::grok_home().join(CURSOR_AUTH_FILE_NAME)
}

pub fn load_credentials() -> io::Result<Option<CursorCredentials>> {
    load_credentials_at(&auth_file_path())
}

pub fn is_logged_in() -> bool {
    process_auth_token_override().is_some()
        || env_api_key().is_some()
        || load_credentials().ok().flatten().is_some()
}

/// Return an auth-required error that callers can distinguish from an ordinary
/// session-load failure without parsing user-facing prose.
pub fn auth_required_error() -> anyhow::Error {
    anyhow!("{CURSOR_AUTH_REQUIRED_ERROR_KIND}: {CURSOR_AUTH_REQUIRED_MESSAGE}")
}

pub async fn fresh_credentials() -> Result<Option<CursorCredentials>> {
    resolve_credentials(false).await
}

pub(crate) async fn force_refresh() -> Result<Option<CursorCredentials>> {
    resolve_credentials(true).await
}

async fn resolve_credentials(force: bool) -> Result<Option<CursorCredentials>> {
    if let Some(token) = process_auth_token_override() {
        return Ok(Some(credentials_from_access_token(
            token,
            None,
            env_api_key(),
        )));
    }

    match refresh_at(&auth_file_path(), &auth_base_url(), force).await {
        Ok(Some(creds)) => return Ok(Some(creds)),
        Ok(None) => {}
        Err(error) => {
            // Fall through to API-key exchange when the file is unusable and an
            // API key is configured; otherwise surface the refresh failure.
            if env_api_key().is_none() {
                return Err(error);
            }
            tracing::debug!(%error, "Cursor file refresh failed; trying API key exchange");
        }
    }

    if let Some(api_key) = env_api_key() {
        let stored = exchange_api_key(&auth_base_url(), &api_key).await?;
        return Ok(Some(credentials_from_stored(&stored)));
    }

    Ok(None)
}

pub async fn run_cli_login(no_browser: bool) -> Result<CursorAccountSummary> {
    let credentials = run_login_at(&auth_file_path(), &auth_base_url(), true, !no_browser).await?;
    Ok(CursorAccountSummary::from(&credentials))
}

/// Browser OAuth for the pager. Unlike the CLI entrypoint, this does not write
/// directly to stderr and disturb the terminal alternate screen.
pub async fn run_tui_login() -> Result<CursorAccountSummary> {
    let credentials = run_login_at(&auth_file_path(), &auth_base_url(), false, true).await?;
    Ok(CursorAccountSummary::from(&credentials))
}

pub async fn run_cli_logout() -> Result<bool> {
    let path = auth_file_path();
    let lock_path = path.clone();
    let _lock = tokio::task::spawn_blocking(move || acquire_auth_lock(&lock_path)).await??;
    let removed = match std::fs::remove_file(&path) {
        Ok(()) => true,
        Err(error) if error.kind() == io::ErrorKind::NotFound => false,
        Err(error) => return Err(error.into()),
    };
    Ok(removed)
}

/// Starts one process-wide refresh loop. The immediate pass makes a cached
/// Cursor token fresh before the first model request; later passes keep the
/// live bearer resolver current for long-running sessions.
pub fn start_proactive_refresh(cancel: tokio_util::sync::CancellationToken) {
    static STARTED: AtomicBool = AtomicBool::new(false);
    if STARTED.swap(true, Ordering::AcqRel) {
        return;
    }
    tokio::spawn(async move {
        loop {
            if let Err(error) = fresh_credentials().await {
                tracing::debug!(%error, "Cursor proactive OAuth refresh skipped");
            }
            tokio::select! {
                () = cancel.cancelled() => break,
                () = tokio::time::sleep(Duration::from_secs(4 * 60)) => {}
            }
        }
    });
}

fn load_store_at(path: &Path) -> io::Result<Option<StoredCursorAuth>> {
    crate::util::secure_file::ensure_owner_only_permissions(path)?;
    let contents = match std::fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    if contents.trim().is_empty() {
        return Ok(None);
    }
    let store: StoredCursorAuth = serde_json::from_str(&contents)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if store.access_token.trim().is_empty() {
        return Ok(None);
    }
    Ok(Some(store))
}

fn load_credentials_at(path: &Path) -> io::Result<Option<CursorCredentials>> {
    Ok(load_store_at(path)?.map(|store| credentials_from_stored(&store)))
}

fn credentials_from_stored(store: &StoredCursorAuth) -> CursorCredentials {
    credentials_from_access_token(
        store.access_token.clone(),
        store.refresh_token.clone(),
        store.api_key.clone(),
    )
}

fn credentials_from_access_token(
    access_token: String,
    _refresh_token: Option<String>,
    api_key: Option<String>,
) -> CursorCredentials {
    let claims = jwt_claims(&access_token);
    let email = claims
        .as_ref()
        .and_then(|claims| claims.get("email"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned);
    let user_id = claims
        .as_ref()
        .and_then(|claims| claims.get("sub"))
        .and_then(Value::as_str)
        .map(cursor_user_id_from_sub);
    CursorCredentials {
        access_token,
        api_key,
        email,
        user_id,
    }
}

fn cursor_user_id_from_sub(sub: &str) -> String {
    // Cursor JWTs often use `auth0|user_…` (or similar) subjects; the trailing
    // segment is the stable user id used by Cursor APIs.
    sub.rsplit('|').next().unwrap_or(sub).trim().to_owned()
}

fn jwt_claims(token: &str) -> Option<Value> {
    let payload = token.split('.').nth(1)?;
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload)
        .ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn jwt_expiration(token: &str) -> Option<DateTime<Utc>> {
    jwt_claims(token)?
        .get("exp")?
        .as_i64()
        .and_then(|timestamp| DateTime::from_timestamp(timestamp, 0))
}

/// Fail-closed freshness check: refresh when `now > exp - 5 minutes`, and treat
/// an unparseable token as expired so it refreshes instead of failing upstream.
fn access_token_is_fresh(access_token: &str) -> bool {
    match jwt_expiration(access_token) {
        Some(expires_at) => expires_at.timestamp() > Utc::now().timestamp() + REFRESH_WINDOW_SECS,
        None => false,
    }
}

fn save_store_at(path: &Path, store: &StoredCursorAuth) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temp = path.with_extension(format!("json.{}.tmp", std::process::id()));
    let file = crate::util::secure_file::open_secure_file(&temp)?;
    let mut writer = io::BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, store)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer
        .into_inner()
        .map_err(|error| error.into_error())?
        .sync_all()?;
    #[cfg(windows)]
    crate::util::secure_file::set_windows_secure_permissions(&temp)?;
    #[cfg(windows)]
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    std::fs::rename(&temp, path)?;
    crate::util::secure_file::ensure_owner_only_permissions(path)?;
    Ok(())
}

fn acquire_auth_lock(path: &Path) -> io::Result<File> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let lock_path = path.with_file_name(format!("{CURSOR_AUTH_FILE_NAME}.lock"));
    let mut options = OpenOptions::new();
    options.create(true).read(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    let file = options.open(lock_path)?;
    fs2::FileExt::lock_exclusive(&file)?;
    Ok(file)
}

fn generate_pkce() -> (String, String) {
    let mut random_bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut random_bytes);
    let verifier = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(random_bytes);
    let challenge = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(Sha256::digest(verifier.as_bytes()));
    (verifier, challenge)
}

fn build_login_url(challenge: &str, uuid: &str) -> String {
    format!("{CURSOR_LOGIN_URL}?challenge={challenge}&uuid={uuid}&mode=login&redirectTarget=cli")
}

fn parse_token_response(value: &Value) -> Option<StoredCursorAuth> {
    // An empty accessToken is not a usable credential; treat a malformed success
    // response as invalid rather than persisting a broken token.
    let access_token = value.get("accessToken")?.as_str()?;
    if access_token.is_empty() {
        return None;
    }
    Some(StoredCursorAuth {
        access_token: access_token.to_owned(),
        refresh_token: value
            .get("refreshToken")
            .and_then(Value::as_str)
            .map(str::to_owned),
        api_key: value
            .get("apiKey")
            .and_then(Value::as_str)
            .map(str::to_owned),
    })
}

/// Per RFC 6749 §6, a refresh response MAY omit a new refresh token; keep the
/// previous one so the next refresh still works.
fn merge_refreshed_store(previous: &StoredCursorAuth, mut refreshed: StoredCursorAuth) -> StoredCursorAuth {
    if refreshed.refresh_token.is_none() {
        refreshed.refresh_token = previous.refresh_token.clone();
    }
    if refreshed.api_key.is_none() {
        refreshed.api_key = previous.api_key.clone();
    }
    refreshed
}

fn process_auth_token_override() -> Option<String> {
    static TOKEN: OnceLock<Option<String>> = OnceLock::new();
    TOKEN
        .get_or_init(|| {
            if let Some(token) = std::env::var("GROK_CURSOR_AUTH_TOKEN")
                .ok()
                .filter(|token| !token.trim().is_empty())
            {
                return Some(token);
            }
            let token = std::env::var("CURSOR_AUTH_TOKEN")
                .ok()
                .filter(|token| !token.trim().is_empty())?;
            tracing::warn!(
                "using CURSOR_AUTH_TOKEN from the environment (not GROK_CURSOR_AUTH_TOKEN); \
                 this bypasses the stored Cursor credential file and its token refresh"
            );
            Some(token)
        })
        .clone()
}

fn env_api_key() -> Option<String> {
    for name in ["GROK_CURSOR_API_KEY", "CURSOR_API_KEY"] {
        if let Ok(value) = std::env::var(name) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_owned());
            }
        }
    }
    None
}

async fn run_login_at(
    path: &Path,
    base_url: &str,
    announce: bool,
    open_browser: bool,
) -> Result<CursorCredentials> {
    let (verifier, challenge) = generate_pkce();
    let uuid = uuid::Uuid::new_v4().to_string();
    let login_url = build_login_url(&challenge, &uuid);

    if announce {
        eprintln!();
        eprintln!("Signing in to Cursor...");
        eprintln!("Open this URL if your browser does not open automatically:");
        eprintln!("  {login_url}");
    }

    if open_browser {
        let open_url = login_url.clone();
        let browser_result = tokio::task::spawn_blocking(move || webbrowser::open(&open_url)).await;
        if !announce {
            match browser_result {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    return Err(anyhow!(
                        "could not open a browser for Cursor login: {error}. Run `grok login --cursor` instead"
                    ));
                }
                Err(error) => {
                    return Err(anyhow!(
                        "could not launch Cursor browser login: {error}. Run `grok login --cursor` instead"
                    ));
                }
            }
        } else if let Ok(Err(error)) = &browser_result {
            eprintln!("Could not open browser automatically: {error}");
        }
    }

    let client = reqwest::Client::new();
    let tokens = poll_for_tokens(&client, base_url, &uuid, &verifier).await?;
    let path_owned = path.to_path_buf();
    let tokens_for_write = tokens.clone();
    tokio::task::spawn_blocking(move || {
        let _lock = acquire_auth_lock(&path_owned)?;
        save_store_at(&path_owned, &tokens_for_write)
    })
    .await
    .map_err(|error| anyhow!("Cursor auth write task failed: {error}"))?
    .with_context(|| format!("failed to write Cursor credentials to {}", path.display()))?;

    if announce {
        eprintln!(
            "Login successful. Credentials saved to {}",
            path.display()
        );
    }
    Ok(credentials_from_stored(&tokens))
}

async fn poll_for_tokens(
    client: &reqwest::Client,
    base_url: &str,
    uuid: &str,
    verifier: &str,
) -> Result<StoredCursorAuth> {
    let mut delay = Duration::from_secs(1);
    let mut last_error: Option<reqwest::Error> = None;
    for _ in 0..POLL_MAX_ATTEMPTS {
        let response = match client
            .get(format!(
                "{}/auth/poll?uuid={uuid}&verifier={verifier}",
                base_url.trim_end_matches('/')
            ))
            .header("content-type", "application/json")
            .timeout(AUTH_REQUEST_TIMEOUT)
            .send()
            .await
        {
            Ok(response) => response,
            Err(error) => {
                last_error = Some(error);
                tokio::time::sleep(delay).await;
                delay = (delay.mul_f32(1.2)).min(Duration::from_secs(10));
                continue;
            }
        };
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            tokio::time::sleep(delay).await;
            delay = (delay.mul_f32(1.2)).min(Duration::from_secs(10));
            continue;
        }
        let status = response.status();
        let text = response
            .text()
            .await
            .context("invalid Cursor poll response")?;
        let value: Value = serde_json::from_str(&text).context("invalid Cursor poll response")?;
        if !status.is_success() {
            bail!("Cursor login poll failed (HTTP {status}): {value}");
        }
        return parse_token_response(&value)
            .ok_or_else(|| anyhow!("Cursor login response missing accessToken"));
    }
    match last_error {
        Some(error) => Err(anyhow::Error::new(error).context(
            "Cursor login timed out after repeated network errors; \
             run `grok login --cursor` to try again",
        )),
        None => bail!("Cursor login timed out; run `grok login --cursor` to try again"),
    }
}

async fn refresh_at(path: &Path, base_url: &str, force: bool) -> Result<Option<CursorCredentials>> {
    let Some(initial) = load_store_at(path)? else {
        return Ok(None);
    };
    if !force && access_token_is_fresh(&initial.access_token) {
        return Ok(Some(credentials_from_stored(&initial)));
    }

    let path_owned = path.to_path_buf();
    let _lock = tokio::task::spawn_blocking(move || acquire_auth_lock(&path_owned)).await??;
    let Some(store) = load_store_at(path)? else {
        return Ok(None);
    };
    if !force && access_token_is_fresh(&store.access_token) {
        return Ok(Some(credentials_from_stored(&store)));
    }

    let refresh_token = store
        .refresh_token
        .as_deref()
        .filter(|token| !token.trim().is_empty())
        .ok_or_else(|| {
            anyhow!("Cursor access token expired; run `grok login --cursor`")
        })?
        .to_owned();

    let client = reqwest::Client::new();
    let refreshed = refresh_tokens(&client, base_url, &refresh_token).await?;
    let merged = merge_refreshed_store(&store, refreshed);
    save_store_at(path, &merged)?;
    tracing::info!("refreshed Cursor OAuth access token");
    Ok(Some(credentials_from_stored(&merged)))
}

async fn refresh_tokens(
    client: &reqwest::Client,
    base_url: &str,
    refresh_token: &str,
) -> Result<StoredCursorAuth> {
    let response = client
        .post(format!("{}/auth/refresh", base_url.trim_end_matches('/')))
        .bearer_auth(refresh_token)
        .header("content-type", "application/json")
        .body("{}")
        .timeout(AUTH_REQUEST_TIMEOUT)
        .send()
        .await
        .context("failed to refresh Cursor auth; run `grok login --cursor`")?;
    if !response.status().is_success() {
        bail!(
            "Cursor token refresh failed (HTTP {}); run `grok login --cursor`",
            response.status()
        );
    }
    let text = response
        .text()
        .await
        .context("invalid Cursor refresh response; run `grok login --cursor`")?;
    let value: Value = serde_json::from_str(&text)
        .context("invalid Cursor refresh response; run `grok login --cursor`")?;
    parse_token_response(&value)
        .ok_or_else(|| anyhow!("invalid Cursor refresh response; run `grok login --cursor`"))
}

async fn exchange_api_key(base_url: &str, api_key: &str) -> Result<StoredCursorAuth> {
    let response = reqwest::Client::new()
        .post(format!(
            "{}/auth/exchange_user_api_key",
            base_url.trim_end_matches('/')
        ))
        .bearer_auth(api_key)
        .header("content-type", "application/json")
        .body("{}")
        .timeout(AUTH_REQUEST_TIMEOUT)
        .send()
        .await
        .context("failed to exchange Cursor API key")?;
    if !response.status().is_success() {
        bail!(
            "Cursor API key exchange failed (HTTP {}); check CURSOR_API_KEY",
            response.status()
        );
    }
    let text = response
        .text()
        .await
        .context("invalid Cursor API key exchange response")?;
    let value: Value =
        serde_json::from_str(&text).context("invalid Cursor API key exchange response")?;
    let mut stored = parse_token_response(&value)
        .ok_or_else(|| anyhow!("Cursor API key exchange response missing accessToken"))?;
    if stored.api_key.is_none() {
        stored.api_key = Some(api_key.to_owned());
    }
    Ok(stored)
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use serde_json::json;

    fn jwt(payload: serde_json::Value) -> String {
        let header = URL_SAFE_NO_PAD.encode(b"{}");
        let payload = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&payload).unwrap());
        format!("{header}.{payload}.signature")
    }

    #[test]
    fn parse_token_response_accepts_camel_case() {
        let auth = parse_token_response(&json!({
            "accessToken": "access",
            "refreshToken": "refresh",
            "apiKey": "key"
        }))
        .unwrap();
        assert_eq!(auth.access_token, "access");
        assert_eq!(auth.refresh_token.as_deref(), Some("refresh"));
        assert_eq!(auth.api_key.as_deref(), Some("key"));
    }

    #[test]
    fn parse_token_response_rejects_empty_access_token() {
        assert!(parse_token_response(&json!({"refreshToken": "refresh"})).is_none());
        assert!(parse_token_response(&json!({"accessToken": ""})).is_none());
    }

    #[test]
    fn refresh_preserves_refresh_token_when_omitted() {
        let previous = StoredCursorAuth {
            access_token: "old-access".to_owned(),
            refresh_token: Some("old-refresh".to_owned()),
            api_key: Some("key".to_owned()),
        };
        let refreshed = StoredCursorAuth {
            access_token: "new-access".to_owned(),
            refresh_token: None,
            api_key: None,
        };
        let merged = merge_refreshed_store(&previous, refreshed);
        assert_eq!(merged.access_token, "new-access");
        assert_eq!(merged.refresh_token.as_deref(), Some("old-refresh"));
        assert_eq!(merged.api_key.as_deref(), Some("key"));
    }

    #[test]
    fn refresh_rotates_refresh_token_when_present() {
        let previous = StoredCursorAuth {
            access_token: "old-access".to_owned(),
            refresh_token: Some("old-refresh".to_owned()),
            api_key: None,
        };
        let refreshed = StoredCursorAuth {
            access_token: "new-access".to_owned(),
            refresh_token: Some("new-refresh".to_owned()),
            api_key: None,
        };
        let merged = merge_refreshed_store(&previous, refreshed);
        assert_eq!(merged.refresh_token.as_deref(), Some("new-refresh"));
    }

    #[test]
    fn pkce_challenge_is_correct_sha256() {
        let (verifier, challenge) = generate_pkce();
        let expected = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
        assert_eq!(challenge, expected);

        // Fixed vector: challenge is base64url(SHA256(verifier ASCII bytes)).
        let verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
        let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
        assert_eq!(challenge, "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM");
    }

    #[test]
    fn auth_file_path_uses_grok_home() {
        let path = auth_file_path();
        assert_eq!(
            path,
            xai_grok_config::grok_home().join(CURSOR_AUTH_FILE_NAME)
        );
        assert_eq!(
            path.file_name().and_then(|name| name.to_str()),
            Some(CURSOR_AUTH_FILE_NAME)
        );
    }

    #[test]
    fn access_token_without_exp_is_not_fresh() {
        assert!(!access_token_is_fresh("not-a-jwt"));
        let no_exp = jwt(json!({"sub": "user"}));
        assert!(!access_token_is_fresh(&no_exp));
    }

    #[test]
    fn credentials_extract_email_and_user_id_from_jwt() {
        let token = jwt(json!({
            "email": "dev@example.com",
            "sub": "auth0|user_abc",
            "exp": 4_102_444_800_i64
        }));
        let creds = credentials_from_access_token(token, None, None);
        assert_eq!(creds.email.as_deref(), Some("dev@example.com"));
        assert_eq!(creds.user_id.as_deref(), Some("user_abc"));
    }

    #[test]
    fn storage_is_owner_only_and_isolated_from_xai_auth() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(CURSOR_AUTH_FILE_NAME);
        let store = StoredCursorAuth {
            access_token: "access".to_owned(),
            refresh_token: Some("refresh".to_owned()),
            api_key: None,
        };
        save_store_at(&path, &store).unwrap();
        assert!(!dir.path().join("auth.json").exists());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        let loaded = load_store_at(&path).unwrap().unwrap();
        assert_eq!(loaded, store);
    }

    #[test]
    fn is_trusted_agent_base_url_matches_configured_endpoint() {
        assert!(is_trusted_agent_base_url(""));
        assert!(is_trusted_agent_base_url(CURSOR_AGENT_BASE_URL));
        assert!(is_trusted_agent_base_url(&format!("{CURSOR_AGENT_BASE_URL}/")));
        assert!(!is_trusted_agent_base_url("https://evil.example.com"));
    }

    #[test]
    fn bearer_resolver_returns_seed_token() {
        let resolver = CursorBearerResolver::from_token("seed-token".to_owned());
        // Without a file/env override this process can see, the seed is used.
        if process_auth_token_override().is_none() && load_credentials().ok().flatten().is_none() {
            assert_eq!(
                xai_grok_sampler::BearerResolver::current_bearer(&resolver).as_deref(),
                Some("seed-token")
            );
        }
    }

    #[tokio::test]
    async fn refresh_http_preserves_refresh_token_when_response_omits_it() {
        use axum::Router;
        use axum::extract::State;
        use axum::routing::post;
        use std::sync::Arc;
        use std::sync::atomic::AtomicUsize;
        use tokio::net::TcpListener;

        async fn handler(
            State(calls): State<Arc<AtomicUsize>>,
        ) -> axum::Json<serde_json::Value> {
            calls.fetch_add(1, Ordering::SeqCst);
            axum::Json(json!({
                "accessToken": jwt(json!({"exp": 4_102_444_800_i64}))
            }))
        }

        let calls = Arc::new(AtomicUsize::new(0));
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new()
            .route("/auth/refresh", post(handler))
            .with_state(calls.clone());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(CURSOR_AUTH_FILE_NAME);
        save_store_at(
            &path,
            &StoredCursorAuth {
                access_token: jwt(json!({"exp": 0_i64})),
                refresh_token: Some("old-refresh".to_owned()),
                api_key: Some("key".to_owned()),
            },
        )
        .unwrap();

        let base = format!("http://{address}");
        let creds = refresh_at(&path, &base, true).await.unwrap().unwrap();
        assert!(access_token_is_fresh(&creds.access_token));
        assert_eq!(creds.api_key.as_deref(), Some("key"));

        let stored = load_store_at(&path).unwrap().unwrap();
        assert_eq!(stored.refresh_token.as_deref(), Some("old-refresh"));
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        server.abort();
    }
}
