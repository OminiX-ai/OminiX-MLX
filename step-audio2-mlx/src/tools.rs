//! Tool calling support for Step-Audio 2
//!
//! Step-Audio 2 mini supports tool calling for extended capabilities.
//! The primary tool is web search, allowing the model to access real-time information.
//!
//! ## Tool Call Format
//!
//! The model outputs tool calls in a structured format:
//! ```text
//! <tool_call>
//! {"name": "web_search", "parameters": {"query": "latest news"}}
//! </tool_call>
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use step_audio2_mlx::tools::{ToolManager, WebSearchTool};
//!
//! let mut manager = ToolManager::new();
//! manager.register(Box::new(WebSearchTool::new(Some("api_key".into()))));
//!
//! // Parse tool calls from model output
//! if let Some(call) = manager.parse_tool_call(&model_output) {
//!     let result = manager.execute(&call)?;
//!     // Feed result back to model
//! }
//! ```

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Tool call markers
pub mod markers {
    pub const TOOL_CALL_START: &str = "<tool_call>";
    pub const TOOL_CALL_END: &str = "</tool_call>";
    pub const TOOL_RESULT_START: &str = "<tool_result>";
    pub const TOOL_RESULT_END: &str = "</tool_result>";
}

/// Tool trait for extensible tool support
pub trait Tool: Send + Sync {
    /// Tool name (must be unique)
    fn name(&self) -> &str;

    /// Tool description for the model
    fn description(&self) -> &str;

    /// Parameter schema in JSON format
    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {}
        })
    }

    /// Execute tool with parameters
    fn execute(&self, params: &Value) -> Result<String>;
}

/// Tool call parsed from model output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Tool name
    pub name: String,
    /// Tool parameters as JSON
    pub parameters: Value,
}

impl ToolCall {
    /// Create a new tool call
    pub fn new(name: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            parameters,
        }
    }

    /// Create a web search tool call
    pub fn web_search(query: impl Into<String>) -> Self {
        Self::new(
            "web_search",
            serde_json::json!({ "query": query.into() }),
        )
    }
}

/// Tool execution result
#[derive(Debug, Clone)]
pub struct ToolResult {
    /// Tool name that was called
    pub tool_name: String,
    /// Result content
    pub content: String,
    /// Whether execution was successful
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

impl ToolResult {
    /// Create a successful result
    pub fn success(tool_name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_name: tool_name.into(),
            content: content.into(),
            success: true,
            error: None,
        }
    }

    /// Create a failed result
    pub fn failure(tool_name: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            tool_name: tool_name.into(),
            content: String::new(),
            success: false,
            error: Some(error.into()),
        }
    }

    /// Format result for model consumption
    pub fn format_for_model(&self) -> String {
        if self.success {
            format!(
                "{}\n{}\n{}",
                markers::TOOL_RESULT_START,
                self.content,
                markers::TOOL_RESULT_END
            )
        } else {
            format!(
                "{}\nError: {}\n{}",
                markers::TOOL_RESULT_START,
                self.error.as_deref().unwrap_or("Unknown error"),
                markers::TOOL_RESULT_END
            )
        }
    }
}

/// Web search tool
///
/// Searches the web for information. Can use different search backends:
/// - Mock (default): Returns placeholder results
/// - Custom: Uses provided search function
pub struct WebSearchTool {
    /// API key for search service (optional)
    pub api_key: Option<String>,
    /// Maximum results to return
    pub max_results: usize,
    /// Custom search function (for testing or custom backends)
    search_fn: Option<Box<dyn Fn(&str) -> Result<Vec<SearchResult>> + Send + Sync>>,
}

/// Search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Result title
    pub title: String,
    /// Result snippet/description
    pub snippet: String,
    /// Source URL
    pub url: String,
}

impl WebSearchTool {
    /// Create a new web search tool
    pub fn new(api_key: Option<String>) -> Self {
        Self {
            api_key,
            max_results: 5,
            search_fn: None,
        }
    }

    /// Create with custom search function
    pub fn with_search_fn<F>(api_key: Option<String>, search_fn: F) -> Self
    where
        F: Fn(&str) -> Result<Vec<SearchResult>> + Send + Sync + 'static,
    {
        Self {
            api_key,
            max_results: 5,
            search_fn: Some(Box::new(search_fn)),
        }
    }

    /// Set maximum results
    pub fn max_results(mut self, max: usize) -> Self {
        self.max_results = max;
        self
    }

    /// Perform the search
    fn search(&self, query: &str) -> Result<Vec<SearchResult>> {
        if let Some(ref search_fn) = self.search_fn {
            return search_fn(query);
        }

        // Mock search results for demonstration
        // In production, this would call a real search API
        Ok(vec![
            SearchResult {
                title: format!("Result 1 for: {}", query),
                snippet: format!("This is a search result snippet for the query '{}'.", query),
                url: format!("https://example.com/search?q={}", query.replace(' ', "+")),
            },
            SearchResult {
                title: format!("Result 2 for: {}", query),
                snippet: "Additional relevant information found.".to_string(),
                url: "https://example.com/result2".to_string(),
            },
        ])
    }
}

impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the web for current information. Use this when you need up-to-date facts, news, or information not in your training data."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        })
    }

    fn execute(&self, params: &Value) -> Result<String> {
        let query = params["query"]
            .as_str()
            .ok_or_else(|| Error::Tool("Missing 'query' parameter".into()))?;

        if query.trim().is_empty() {
            return Err(Error::Tool("Search query cannot be empty".into()));
        }

        let results = self.search(query)?;

        // Format results for the model
        let mut output = format!("Search results for \"{}\":\n\n", query);
        for (i, result) in results.iter().take(self.max_results).enumerate() {
            output.push_str(&format!(
                "{}. {}\n   {}\n   Source: {}\n\n",
                i + 1,
                result.title,
                result.snippet,
                result.url
            ));
        }

        Ok(output)
    }
}

/// Calculator tool for basic math operations
pub struct CalculatorTool;

impl CalculatorTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CalculatorTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for CalculatorTool {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Perform basic mathematical calculations. Supports +, -, *, /, and parentheses."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 3 * 4')"
                }
            },
            "required": ["expression"]
        })
    }

    fn execute(&self, params: &Value) -> Result<String> {
        let expr = params["expression"]
            .as_str()
            .ok_or_else(|| Error::Tool("Missing 'expression' parameter".into()))?;

        // Simple expression evaluator (basic operations only)
        let result = evaluate_simple_expression(expr)?;
        Ok(format!("{} = {}", expr, result))
    }
}

/// Simple expression evaluator (supports +, -, *, /)
fn evaluate_simple_expression(expr: &str) -> Result<f64> {
    // Remove whitespace
    let expr: String = expr.chars().filter(|c| !c.is_whitespace()).collect();

    // Parse and evaluate
    let mut chars = expr.chars().peekable();
    parse_expression(&mut chars)
}

fn parse_expression(chars: &mut std::iter::Peekable<std::str::Chars>) -> Result<f64> {
    let mut result = parse_term(chars)?;

    while let Some(&c) = chars.peek() {
        match c {
            '+' => {
                chars.next();
                result += parse_term(chars)?;
            }
            '-' => {
                chars.next();
                result -= parse_term(chars)?;
            }
            _ => break,
        }
    }

    Ok(result)
}

fn parse_term(chars: &mut std::iter::Peekable<std::str::Chars>) -> Result<f64> {
    let mut result = parse_factor(chars)?;

    while let Some(&c) = chars.peek() {
        match c {
            '*' => {
                chars.next();
                result *= parse_factor(chars)?;
            }
            '/' => {
                chars.next();
                let divisor = parse_factor(chars)?;
                if divisor == 0.0 {
                    return Err(Error::Tool("Division by zero".into()));
                }
                result /= divisor;
            }
            _ => break,
        }
    }

    Ok(result)
}

fn parse_factor(chars: &mut std::iter::Peekable<std::str::Chars>) -> Result<f64> {
    // Handle parentheses
    if chars.peek() == Some(&'(') {
        chars.next(); // consume '('
        let result = parse_expression(chars)?;
        if chars.next() != Some(')') {
            return Err(Error::Tool("Mismatched parentheses".into()));
        }
        return Ok(result);
    }

    // Handle negative numbers
    let negative = if chars.peek() == Some(&'-') {
        chars.next();
        true
    } else {
        false
    };

    // Parse number
    let mut num_str = String::new();
    while let Some(&c) = chars.peek() {
        if c.is_ascii_digit() || c == '.' {
            num_str.push(c);
            chars.next();
        } else {
            break;
        }
    }

    if num_str.is_empty() {
        return Err(Error::Tool("Invalid expression".into()));
    }

    let num: f64 = num_str
        .parse()
        .map_err(|_| Error::Tool(format!("Invalid number: {}", num_str)))?;

    Ok(if negative { -num } else { num })
}

/// Tool manager for registering and executing tools
pub struct ToolManager {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolManager {
    /// Create a new tool manager
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Create with default tools (web search, calculator)
    pub fn with_defaults() -> Self {
        let mut manager = Self::new();
        manager.register(Box::new(WebSearchTool::new(None)));
        manager.register(Box::new(CalculatorTool::new()));
        manager
    }

    /// Register a tool
    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Get a tool by name
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    /// Get all registered tool names
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Generate tool description for model prompt
    pub fn generate_tool_prompt(&self) -> String {
        let mut prompt = String::from("Available tools:\n\n");

        for tool in self.tools.values() {
            prompt.push_str(&format!(
                "- {}: {}\n  Parameters: {}\n\n",
                tool.name(),
                tool.description(),
                tool.parameters_schema()
            ));
        }

        prompt.push_str(&format!(
            "To use a tool, output:\n{}\n{{\"name\": \"tool_name\", \"parameters\": {{...}}}}\n{}\n",
            markers::TOOL_CALL_START,
            markers::TOOL_CALL_END
        ));

        prompt
    }

    /// Parse tool call from model output
    ///
    /// Returns the first tool call found in the output, if any.
    pub fn parse_tool_call(&self, output: &str) -> Option<ToolCall> {
        // Find tool call markers
        let start_idx = output.find(markers::TOOL_CALL_START)?;
        let end_idx = output.find(markers::TOOL_CALL_END)?;

        if start_idx >= end_idx {
            return None;
        }

        // Extract JSON between markers
        let json_start = start_idx + markers::TOOL_CALL_START.len();
        let json_str = output[json_start..end_idx].trim();

        // Parse JSON
        let call: ToolCall = serde_json::from_str(json_str).ok()?;

        // Validate tool exists
        if !self.tools.contains_key(&call.name) {
            return None;
        }

        Some(call)
    }

    /// Parse all tool calls from model output
    pub fn parse_all_tool_calls(&self, output: &str) -> Vec<ToolCall> {
        let mut calls = Vec::new();
        let mut remaining = output;

        while let Some(start_idx) = remaining.find(markers::TOOL_CALL_START) {
            let after_start = &remaining[start_idx + markers::TOOL_CALL_START.len()..];

            if let Some(end_idx) = after_start.find(markers::TOOL_CALL_END) {
                let json_str = after_start[..end_idx].trim();

                if let Ok(call) = serde_json::from_str::<ToolCall>(json_str) {
                    if self.tools.contains_key(&call.name) {
                        calls.push(call);
                    }
                }

                remaining = &after_start[end_idx + markers::TOOL_CALL_END.len()..];
            } else {
                break;
            }
        }

        calls
    }

    /// Check if output contains a tool call
    pub fn has_tool_call(&self, output: &str) -> bool {
        output.contains(markers::TOOL_CALL_START) && output.contains(markers::TOOL_CALL_END)
    }

    /// Execute a tool call
    pub fn execute(&self, call: &ToolCall) -> ToolResult {
        match self.tools.get(&call.name) {
            Some(tool) => match tool.execute(&call.parameters) {
                Ok(content) => ToolResult::success(&call.name, content),
                Err(e) => ToolResult::failure(&call.name, e.to_string()),
            },
            None => ToolResult::failure(&call.name, format!("Unknown tool: {}", call.name)),
        }
    }

    /// Execute multiple tool calls
    pub fn execute_all(&self, calls: &[ToolCall]) -> Vec<ToolResult> {
        calls.iter().map(|call| self.execute(call)).collect()
    }
}

impl Default for ToolManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_call_creation() {
        let call = ToolCall::web_search("test query");
        assert_eq!(call.name, "web_search");
        assert_eq!(call.parameters["query"], "test query");
    }

    #[test]
    fn test_web_search_tool() {
        let tool = WebSearchTool::new(None);
        assert_eq!(tool.name(), "web_search");

        let result = tool.execute(&serde_json::json!({"query": "test"}));
        assert!(result.is_ok());
    }

    #[test]
    fn test_web_search_empty_query() {
        let tool = WebSearchTool::new(None);
        let result = tool.execute(&serde_json::json!({"query": ""}));
        assert!(result.is_err());
    }

    #[test]
    fn test_calculator_tool() {
        let tool = CalculatorTool::new();
        assert_eq!(tool.name(), "calculator");

        // Basic operations
        let result = tool.execute(&serde_json::json!({"expression": "2 + 3"}));
        assert!(result.is_ok());
        assert!(result.unwrap().contains("5"));

        let result = tool.execute(&serde_json::json!({"expression": "10 * 5"}));
        assert!(result.unwrap().contains("50"));

        let result = tool.execute(&serde_json::json!({"expression": "100 / 4"}));
        assert!(result.unwrap().contains("25"));
    }

    #[test]
    fn test_calculator_complex() {
        let tool = CalculatorTool::new();

        // Order of operations
        let result = tool.execute(&serde_json::json!({"expression": "2 + 3 * 4"}));
        assert!(result.unwrap().contains("14"));

        // Parentheses
        let result = tool.execute(&serde_json::json!({"expression": "(2 + 3) * 4"}));
        assert!(result.unwrap().contains("20"));
    }

    #[test]
    fn test_calculator_division_by_zero() {
        let tool = CalculatorTool::new();
        let result = tool.execute(&serde_json::json!({"expression": "5 / 0"}));
        assert!(result.is_err());
    }

    #[test]
    fn test_tool_manager_registration() {
        let mut manager = ToolManager::new();
        manager.register(Box::new(WebSearchTool::new(None)));

        assert!(manager.get("web_search").is_some());
        assert!(manager.get("nonexistent").is_none());
    }

    #[test]
    fn test_tool_manager_with_defaults() {
        let manager = ToolManager::with_defaults();
        assert!(manager.get("web_search").is_some());
        assert!(manager.get("calculator").is_some());
    }

    #[test]
    fn test_parse_tool_call() {
        let manager = ToolManager::with_defaults();

        let output = r#"Let me search for that.
<tool_call>
{"name": "web_search", "parameters": {"query": "latest news"}}
</tool_call>
"#;

        let call = manager.parse_tool_call(output);
        assert!(call.is_some());

        let call = call.unwrap();
        assert_eq!(call.name, "web_search");
        assert_eq!(call.parameters["query"], "latest news");
    }

    #[test]
    fn test_parse_multiple_tool_calls() {
        let manager = ToolManager::with_defaults();

        let output = r#"
<tool_call>
{"name": "web_search", "parameters": {"query": "weather"}}
</tool_call>
<tool_call>
{"name": "calculator", "parameters": {"expression": "2+2"}}
</tool_call>
"#;

        let calls = manager.parse_all_tool_calls(output);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "web_search");
        assert_eq!(calls[1].name, "calculator");
    }

    #[test]
    fn test_tool_result_formatting() {
        let result = ToolResult::success("web_search", "Found 5 results");
        let formatted = result.format_for_model();
        assert!(formatted.contains("<tool_result>"));
        assert!(formatted.contains("Found 5 results"));
        assert!(formatted.contains("</tool_result>"));

        let result = ToolResult::failure("web_search", "API error");
        let formatted = result.format_for_model();
        assert!(formatted.contains("Error: API error"));
    }

    #[test]
    fn test_tool_execution() {
        let manager = ToolManager::with_defaults();

        let call = ToolCall::web_search("rust programming");
        let result = manager.execute(&call);

        assert!(result.success);
        assert!(!result.content.is_empty());
    }

    #[test]
    fn test_has_tool_call() {
        let manager = ToolManager::new();

        assert!(manager.has_tool_call("<tool_call>\n{}\n</tool_call>"));
        assert!(!manager.has_tool_call("No tool call here"));
        assert!(!manager.has_tool_call("<tool_call> only start"));
    }

    #[test]
    fn test_generate_tool_prompt() {
        let manager = ToolManager::with_defaults();
        let prompt = manager.generate_tool_prompt();

        assert!(prompt.contains("web_search"));
        assert!(prompt.contains("calculator"));
        assert!(prompt.contains("<tool_call>"));
    }
}
