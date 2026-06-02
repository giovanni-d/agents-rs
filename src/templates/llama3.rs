//! Llama 3 Instruct template.

use crate::{Message, Role};

use super::helpers::{push_text_content, push_tool_result_content};
use super::template::ChatTemplate;

/// Llama 3 Instruct template.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Llama3Template;

impl ChatTemplate for Llama3Template {
    fn format(&self, messages: &[Message]) -> String {
        let mut prompt = String::from("<|begin_of_text|>");
        let has_tools = messages.iter().any(|m| m.role == Role::Tool);

        for msg in messages {
            match &msg.role {
                Role::System => {
                    prompt.push_str(
                        "<|start_header_id|>system<|end_header_id|>\n\n",
                    );
                    if has_tools {
                        prompt.push_str("Environment: ipython\n");
                    }
                    prompt.push_str("Cutting Knowledge Date: December 2023\n");
                    let today = today_date_string();
                    prompt.push_str("Today Date: ");
                    prompt.push_str(&today);
                    prompt.push_str("\n\n");
                    push_text_content(&mut prompt, msg);
                    prompt.push_str("<|eot_id|>");
                }
                Role::User => {
                    prompt.push_str(
                        "<|start_header_id|>user<|end_header_id|>\n\n",
                    );
                    push_text_content(&mut prompt, msg);
                    prompt.push_str("<|eot_id|>");
                }
                Role::Assistant => {
                    prompt.push_str(
                        "<|start_header_id|>assistant<|end_header_id|>\n\n",
                    );
                    push_text_content(&mut prompt, msg);
                    prompt.push_str("<|eot_id|>");
                }
                Role::Tool => {
                    prompt.push_str(
                        "<|start_header_id|>ipython<|end_header_id|>\n\n",
                    );
                    push_tool_result_content(&mut prompt, msg);
                    prompt.push_str("<|eot_id|>");
                }
                Role::Custom(_) => {
                    prompt.push_str(
                        "<|start_header_id|>user<|end_header_id|>\n\n",
                    );
                    push_text_content(&mut prompt, msg);
                    prompt.push_str("<|eot_id|>");
                }
            }
        }

        prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        prompt
    }

    fn stop_tokens(&self) -> &[&str] {
        &["<|eot_id|>"]
    }

    fn format_system_prefix(&self, system: &str) -> Option<String> {
        let mut prompt = String::from(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
        );
        prompt.push_str("Cutting Knowledge Date: December 2023\n");
        prompt.push_str("Today Date: ");
        prompt.push_str(&today_date_string());
        prompt.push_str("\n\n");
        prompt.push_str(system);
        prompt.push_str("<|eot_id|>");
        Some(prompt)
    }
}

fn today_date_string() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let days = (now / 86400) as i64;
    let (year, month, day) = days_to_ymd(days);

    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
        "Nov", "Dec",
    ];

    format!("{:02} {} {}", day, MONTHS[month as usize - 1], year)
}

fn days_to_ymd(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}
