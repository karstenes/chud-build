# Cursor OAuth login

This fork accepts an **isolated Cursor OAuth** account beside xAI, following the
same credential-separation pattern that
[open-grok](https://github.com/mweinbach/open-grok) uses for ChatGPT Codex.

## Commands

| Command | Effect |
| --- | --- |
| `grok login --cursor` | Browser PKCE login (`loginDeepControl` + `/auth/poll`) |
| `NO_OPEN_BROWSER=1 grok login --cursor` | Print the URL; do not open a browser |
| `grok logout --cursor` | Delete `~/.grok/cursor-auth.json` only |
| `grok logout --all` | Cursor logout, then xAI logout |
| `/login cursor` | TUI browser login (does not change xAI ACP auth) |
| `/logout cursor` | TUI Cursor logout |

Bare `grok login` / `grok logout` remain xAI-only.

## Credential isolation

| Store | Path | Owner |
| --- | --- | --- |
| xAI primary | `$GROK_HOME/auth.json` | `AuthManager` |
| Cursor OAuth | `$GROK_HOME/cursor-auth.json` | `cursor_auth` |

Rules:

1. Cursor login/refresh/logout never read or write xAI `auth.json`.
2. Cursor tokens are never installed into the process-wide ACP auth cell.
3. Explicit env overrides (`CURSOR_API_KEY`, `GROK_CURSOR_API_KEY`,
   `CURSOR_AUTH_TOKEN`, `GROK_CURSOR_AUTH_TOKEN`) stay process-local.
4. Trusted agent endpoint override: `GROK_CURSOR_AGENT_BASE_URL`
   (defaults to `https://agentn.global.api5.cursor.sh`).
5. Auth API override: `GROK_CURSOR_AUTH_BASE_URL`
   (defaults to `https://api2.cursor.sh`).

## Implementation map

- `crates/codegen/xai-grok-shell/src/cursor_auth.rs` — store, PKCE login, refresh,
  logout, `CursorBearerResolver`, proactive refresh
- `crates/codegen/xai-grok-shell/src/cursor_models.rs` — catalog entries + live
  `AgentService/GetUsableModels` merge (Bearer), with `api.cursor.com/v0/models`
  and static fallback as backups
- `crates/codegen/xai-grok-sampler/src/cursor_agent.rs` — text-only Connect/HTTP2
  `AgentService/Run` streaming (`ApiBackend::CursorAgent`)
- CLI: `crates/codegen/xai-grok-pager/src/app/cli.rs`,
  `crates/codegen/xai-grok-pager-bin/src/main.rs`
- TUI: `/login cursor`, `/logout cursor`, `Effect::LoginCursor` /
  `Effect::LogoutCursor`

## Sampling

Cursor models in `default_models.json` use `"api_backend": "cursor_agent"` and
`"agent_type": "cursor"`. They appear in the picker only when Cursor is logged
in (`cursor-auth.json` or `CURSOR_API_KEY`). Inference goes to
`https://agentn.global.api5.cursor.sh` over HTTP/2 Connect protobuf
(`AgentService/Run`); tools are deferred (text-only first cut).

Model ids must be AgentService **wire** ids (for example `gpt-5.4-medium`,
`claude-4.6-sonnet-medium`, `composer-2.5`). Short aliases like `sonnet-4.6`
fail Run with Connect `not_found`. Auto is `default`. At startup we prefer
`GetUsableModels` with the OAuth bearer so the picker only offers models this
account can run; Connect `not_found` / `invalid_argument` / auth codes fail
fast (no 15× retry loop).

### Pulling / refreshing Cursor models

There is no separate CLI “pull models” command. The agent pulls automatically:

1. On agent startup when Cursor credentials exist
2. After `/login cursor` / `/logout cursor` (ACP `x.ai/internal/reload_cursor_models`)
3. After a successful xAI catalog refresh

Check `~/.grok/logs/` (or sampling/unified logs) for
`merged Cursor models from AgentService GetUsableModels` vs
`GetUsableModels failed; falling back`. If discovery fails you still get the
bundled fallback list (composer / gpt-5.4-* / claude-4.6-*).

Credentials are resolved via `CursorBearerResolver` and never through xAI
`AuthManager`.

## OAuth contract

1. Generate PKCE verifier/challenge and a login UUID.
2. Open
   `https://cursor.com/loginDeepControl?challenge=…&uuid=…&mode=login&redirectTarget=cli`.
3. Poll `GET https://api2.cursor.sh/auth/poll?uuid=…&verifier=…` until tokens
   arrive (404 = pending).
4. Persist camelCase `{ accessToken, refreshToken?, apiKey? }` with `0600`
   permissions.
5. Refresh with `POST /auth/refresh` using the refresh token as Bearer; keep the
   old refresh token when the response omits a replacement.
