//! `/logout` -- remove auth credentials and return to the login screen.

use crate::app::actions::Action;
use crate::slash::command::{AppCtx, ArgItem, CommandExecCtx, CommandResult, SlashCommand};

pub struct LogoutCommand;

fn provider_items() -> Vec<ArgItem> {
    vec![
        ArgItem {
            display: "xAI Grok".to_owned(),
            match_text: "xai grok".to_owned(),
            insert_text: "xai".to_owned(),
            description: "Sign out of xAI".to_owned(),
        },
        ArgItem {
            display: "Cursor".to_owned(),
            match_text: "cursor".to_owned(),
            insert_text: "cursor".to_owned(),
            description: "Sign out of Cursor (isolated credentials)".to_owned(),
        },
        ArgItem {
            display: "All".to_owned(),
            match_text: "all both".to_owned(),
            insert_text: "all".to_owned(),
            description: "Sign out of xAI and Cursor".to_owned(),
        },
    ]
}

fn provider_action(args: &str) -> Result<Action, String> {
    let provider = args.trim().to_ascii_lowercase();
    match provider.as_str() {
        "" | "xai" | "grok" => Ok(Action::Logout),
        "cursor" => Ok(Action::LogoutCursor),
        "all" => Ok(Action::LogoutAllProviders),
        _ => Err(format!(
            "Unknown provider: {}. Use /logout xai, /logout cursor, or /logout all",
            args.trim()
        )),
    }
}

impl SlashCommand for LogoutCommand {
    fn name(&self) -> &str {
        "logout"
    }

    fn description(&self) -> &str {
        "Sign out of xAI and/or Cursor"
    }

    fn usage(&self) -> &str {
        "/logout [xai|cursor|all]"
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
