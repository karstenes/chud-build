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
- CLI: `crates/codegen/xai-grok-pager/src/app/cli.rs`,
  `crates/codegen/xai-grok-pager-bin/src/main.rs`
- TUI: `/login cursor`, `/logout cursor`, `Effect::LoginCursor` /
  `Effect::LogoutCursor`

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
