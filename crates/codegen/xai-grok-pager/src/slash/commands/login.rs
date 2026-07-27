//! `/login` -- log in or re-authenticate with your account.

use crate::app::actions::Action;
use crate::slash::command::{AppCtx, ArgItem, CommandExecCtx, CommandResult, SlashCommand};

pub struct LoginCommand;

/// Provider choices shared by slash completion and bare `/login` helpers.
pub(crate) fn provider_items() -> Vec<ArgItem> {
    vec![
        ArgItem {
            display: "xAI Grok".to_owned(),
            match_text: "xai grok oauth".to_owned(),
            insert_text: "xai".to_owned(),
            description: "Sign in with xAI".to_owned(),
        },
        ArgItem {
            display: "Cursor".to_owned(),
            match_text: "cursor oauth".to_owned(),
            insert_text: "cursor".to_owned(),
            description: "Connect a Cursor account (isolated OAuth)".to_owned(),
        },
    ]
}

/// Resolve a user-facing provider token to its concrete login action.
pub(crate) fn provider_action(args: &str) -> Result<Action, String> {
    let provider = args.trim().to_ascii_lowercase();
    match provider.as_str() {
        "" | "xai" | "grok" => Ok(Action::Login),
        "cursor" => Ok(Action::LoginCursor),
        _ => Err(format!(
            "Unknown provider: {}. Use /login xai or /login cursor",
            args.trim()
        )),
    }
}

impl SlashCommand for LoginCommand {
    fn name(&self) -> &str {
        "login"
    }

    fn description(&self) -> &str {
        "Connect xAI or Cursor"
    }

    fn usage(&self) -> &str {
        "/login [xai|cursor]"
    }

    fn takes_args(&self) -> bool {
        true
    }

    fn arg_placeholder(&self) -> Option<&str> {
        Some("provider")
    }

    fn suggest_args(&self, _ctx: &AppCtx, _args_query: &str) -> Option<Vec<ArgItem>> {
        Some(provider_items())
    }

    fn run(&self, _ctx: &mut CommandExecCtx, args: &str) -> CommandResult {
        match provider_action(args) {
            Ok(action) => CommandResult::Action(action),
            Err(message) => CommandResult::Error(message),
        }
    }
}
