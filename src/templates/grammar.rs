//! GBNF grammar builder for the simple-JSON tool-call protocol.

use crate::ToolRegistry;

/// Builder for the v0 tool-call GBNF grammar.
pub struct GbnfToolCallGrammar;

impl GbnfToolCallGrammar {
    /// Build a GBNF grammar matching `( <text> | <tool_call> )+` where
    /// each `tool_call` is `{"tool": "<name>", "args": {<schema>}}`.
    /// Returns `None` for an empty registry.
    pub fn from_registry(registry: &ToolRegistry) -> Option<String> {
        if registry.is_empty() {
            return None;
        }

        let tool_calls: Vec<String> = registry
            .tools()
            .map(|tool| {
                let def = tool.definition();
                // The double `{{` / `}}` survive `format!`'s brace-escape
                // and produce single literal braces at runtime.
                format!(
                    "(\"{{\" ws \"\\\"tool\\\":\" ws \"\\\"{name}\\\"\" ws \",\" ws \"\\\"args\\\":\" ws {args} ws \"}}\")",
                    name = def.name,
                    args = def.input.to_gbnf(),
                )
            })
            .collect();

        let tool_call_alt = tool_calls.join(" | ");

        // `text ::= [^{]+` matches any non-empty run not containing the
        // opening brace of a tool call.
        Some(format!(
            "root ::= block+\n\
block ::= text | tool-call\n\
text ::= [^{{]+\n\
tool-call ::= {tool_call_alt}\n\
ws ::= [ \\t\\n]*\n\
string ::= \"\\\"\" ([^\"\\\\] | \"\\\\\" .)* \"\\\"\"\n\
number ::= \"-\"? [0-9]+ (\".\" [0-9]+)? ([eE] [+-]? [0-9]+)?\n\
value ::= string | number | (\"true\" | \"false\") | \"null\" | \
\"[\" ws (value (\",\" ws value)*)? ws \"]\" | \
\"{{\" ws (string \":\" ws value (\",\" ws string \":\" ws value)*)? ws \"}}\"\n",
        ))
    }
}

