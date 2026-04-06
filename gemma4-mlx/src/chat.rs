use std::{collections::HashMap, path::Path};

use mlx_rs::{
    ops::indexing::{IndexOp, NewAxis},
    Array,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokenizers::Tokenizer;

use crate::{load_model, load_tokenizer, Error, Generate, KVCache, Model, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Gemma4Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Gemma4Role {
    fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gemma4ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl Gemma4ToolSpec {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gemma4ToolCall {
    pub name: String,
    pub arguments: Value,
}

impl Gemma4ToolCall {
    pub fn new(name: impl Into<String>, arguments: Value) -> Self {
        Self {
            name: name.into(),
            arguments,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gemma4Message {
    pub role: Gemma4Role,
    pub content: String,
    pub thinking: Option<String>,
    pub tool_calls: Vec<Gemma4ToolCall>,
    pub tool_name: Option<String>,
}

impl Gemma4Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Gemma4Role::System,
            content: content.into(),
            thinking: None,
            tool_calls: Vec::new(),
            tool_name: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Gemma4Role::User,
            content: content.into(),
            thinking: None,
            tool_calls: Vec::new(),
            tool_name: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Gemma4Role::Assistant,
            content: content.into(),
            thinking: None,
            tool_calls: Vec::new(),
            tool_name: None,
        }
    }

    pub fn assistant_with_tool_calls(
        content: impl Into<String>,
        tool_calls: Vec<Gemma4ToolCall>,
    ) -> Self {
        Self {
            role: Gemma4Role::Assistant,
            content: content.into(),
            thinking: None,
            tool_calls,
            tool_name: None,
        }
    }

    pub fn tool(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Gemma4Role::Tool,
            content: content.into(),
            thinking: None,
            tool_calls: Vec::new(),
            tool_name: Some(name.into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gemma4ParsedAssistantResponse {
    pub raw_text: String,
    pub content: String,
    pub thinking: Option<String>,
    pub tool_calls: Vec<Gemma4ToolCall>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4ToolResult {
    pub name: String,
    pub content: Value,
    pub success: bool,
    pub error: Option<String>,
}

impl Gemma4ToolResult {
    pub fn success(name: impl Into<String>, content: Value) -> Self {
        Self {
            name: name.into(),
            content,
            success: true,
            error: None,
        }
    }

    pub fn failure(name: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            content: Value::Null,
            success: false,
            error: Some(error.into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4ChatResponse {
    pub raw_text: String,
    pub text: String,
    pub thinking: Option<String>,
    pub tool_calls: Vec<Gemma4ToolCall>,
    pub tool_results: Vec<Gemma4ToolResult>,
    pub tokens_generated: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct GeneratedAssistantTurn {
    parsed: Gemma4ParsedAssistantResponse,
    tokens_generated: usize,
}

impl GeneratedAssistantTurn {
    fn new(parsed: Gemma4ParsedAssistantResponse, tokens_generated: usize) -> Self {
        Self {
            parsed,
            tokens_generated,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Gemma4Conversation {
    pub messages: Vec<Gemma4Message>,
}

impl Gemma4Conversation {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_user(&mut self, content: impl Into<String>) {
        self.messages.push(Gemma4Message::user(content));
    }

    pub fn add_assistant_message(&mut self, message: Gemma4Message) {
        self.messages.push(message);
    }

    pub fn add_tool_result(&mut self, result: &Gemma4ToolResult) {
        let content = if result.success {
            result.content.clone()
        } else {
            serde_json::json!({ "error": result.error.clone().unwrap_or_default() })
        };
        self.messages
            .push(Gemma4Message::tool(&result.name, content.to_string()));
    }
}

#[derive(Debug, Clone)]
pub struct Gemma4ChatConfig {
    pub temperature: f32,
    pub max_new_tokens: usize,
    pub max_tool_iterations: usize,
}

impl Default for Gemma4ChatConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            max_new_tokens: 2048,
            max_tool_iterations: 4,
        }
    }
}

pub trait Gemma4Tool: Send + Sync {
    fn spec(&self) -> Gemma4ToolSpec;
    fn execute(&self, arguments: &Value) -> Result<Value>;
}

pub struct Gemma4FunctionTool<F>
where
    F: Fn(&Value) -> Result<Value> + Send + Sync + 'static,
{
    spec: Gemma4ToolSpec,
    handler: F,
}

impl<F> Gemma4FunctionTool<F>
where
    F: Fn(&Value) -> Result<Value> + Send + Sync + 'static,
{
    pub fn new(spec: Gemma4ToolSpec, handler: F) -> Self {
        Self { spec, handler }
    }
}

impl<F> Gemma4Tool for Gemma4FunctionTool<F>
where
    F: Fn(&Value) -> Result<Value> + Send + Sync + 'static,
{
    fn spec(&self) -> Gemma4ToolSpec {
        self.spec.clone()
    }

    fn execute(&self, arguments: &Value) -> Result<Value> {
        (self.handler)(arguments)
    }
}

#[derive(Default)]
pub struct Gemma4ToolRegistry {
    tools: HashMap<String, Box<dyn Gemma4Tool>>,
}

impl Gemma4ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<T>(&mut self, tool: T)
    where
        T: Gemma4Tool + 'static,
    {
        let spec = tool.spec();
        self.tools.insert(spec.name.clone(), Box::new(tool));
    }

    pub fn specs(&self) -> Vec<Gemma4ToolSpec> {
        let mut specs: Vec<_> = self.tools.values().map(|tool| tool.spec()).collect();
        specs.sort_by(|left, right| left.name.cmp(&right.name));
        specs
    }

    pub fn execute_call(&self, call: &Gemma4ToolCall) -> Gemma4ToolResult {
        match self.tools.get(&call.name) {
            Some(tool) => match tool.execute(&call.arguments) {
                Ok(content) => Gemma4ToolResult::success(&call.name, content),
                Err(error) => Gemma4ToolResult::failure(&call.name, error.to_string()),
            },
            None => Gemma4ToolResult::failure(&call.name, format!("Unknown tool: {}", call.name)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gemma4SpecialTokens {
    pub bos: String,
    pub eos: String,
    pub turn_start: String,
    pub turn_end: String,
    pub channel_start: String,
    pub channel_end: String,
    pub tool_start: String,
    pub tool_end: String,
    pub tool_call_start: String,
    pub tool_call_end: String,
    pub tool_response_start: String,
    pub tool_response_end: String,
}

impl Default for Gemma4SpecialTokens {
    fn default() -> Self {
        Self {
            bos: "<bos>".to_string(),
            eos: "<eos>".to_string(),
            turn_start: "<|turn>".to_string(),
            turn_end: "<turn|>".to_string(),
            channel_start: "<|channel>".to_string(),
            channel_end: "<channel|>".to_string(),
            tool_start: "<|tool>".to_string(),
            tool_end: "<tool|>".to_string(),
            tool_call_start: "<|tool_call>".to_string(),
            tool_call_end: "<tool_call|>".to_string(),
            tool_response_start: "<|tool_response>".to_string(),
            tool_response_end: "<tool_response|>".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Gemma4ChatTemplate {
    pub tokens: Gemma4SpecialTokens,
    pub default_system_prompt: String,
}

impl Default for Gemma4ChatTemplate {
    fn default() -> Self {
        Self {
            tokens: Gemma4SpecialTokens::default(),
            default_system_prompt: "You are a helpful assistant.".to_string(),
        }
    }
}

impl Gemma4ChatTemplate {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let defaults = Gemma4SpecialTokens::default();
        let tokenizer_config_path = model_dir.as_ref().join("tokenizer_config.json");
        let json = std::fs::read_to_string(tokenizer_config_path)?;
        let config: TokenizerConfigTokens = serde_json::from_str(&json)?;

        Ok(Self {
            tokens: Gemma4SpecialTokens {
                bos: config.bos_token.unwrap_or(defaults.bos),
                eos: config.eos_token.unwrap_or(defaults.eos),
                turn_start: config.sot_token.unwrap_or(defaults.turn_start),
                turn_end: config.eot_token.unwrap_or(defaults.turn_end),
                channel_start: config.soc_token.unwrap_or(defaults.channel_start),
                channel_end: config.eoc_token.unwrap_or(defaults.channel_end),
                tool_start: config.std_token.unwrap_or(defaults.tool_start),
                tool_end: config.etd_token.unwrap_or(defaults.tool_end),
                tool_call_start: config.stc_token.unwrap_or(defaults.tool_call_start),
                tool_call_end: config.etc_token.unwrap_or(defaults.tool_call_end),
                tool_response_start: config.str_token.unwrap_or(defaults.tool_response_start),
                tool_response_end: config.etr_token.unwrap_or(defaults.tool_response_end),
            },
            default_system_prompt: "You are a helpful assistant.".to_string(),
        })
    }

    pub fn render_prompt(
        &self,
        messages: &[Gemma4Message],
        tools: &[Gemma4ToolSpec],
        add_generation_prompt: bool,
    ) -> Result<String> {
        let mut prompt = String::new();
        prompt.push_str(&self.tokens.bos);

        let (system_message, remaining_messages) = match messages.first() {
            Some(message) if message.role == Gemma4Role::System => {
                (message.content.clone(), &messages[1..])
            }
            _ => (self.default_system_prompt.clone(), messages),
        };

        let mut system_body = system_message;
        if !tools.is_empty() {
            if !system_body.is_empty() {
                system_body.push_str("\n\n");
            }
            system_body.push_str(&self.render_tool_instructions(tools)?);
        }
        prompt.push_str(&self.render_turn(Gemma4Role::System, &system_body)?);

        for message in remaining_messages {
            prompt.push_str(&self.render_message(message)?);
        }

        if add_generation_prompt {
            prompt.push_str(&self.tokens.turn_start);
            prompt.push_str(Gemma4Role::Assistant.as_str());
            prompt.push('\n');
        }

        Ok(prompt)
    }

    fn render_tool_instructions(&self, tools: &[Gemma4ToolSpec]) -> Result<String> {
        let mut body = String::from("Available tools:\n");
        for tool in tools {
            body.push_str(&self.tokens.tool_start);
            body.push_str(&serialize_tool_spec(tool)?);
            body.push_str(&self.tokens.tool_end);
            body.push('\n');
        }
        body.push_str(
            "When you call a tool, respond with one or more tool calls formatted exactly as:\n",
        );
        body.push_str(&self.tokens.tool_call_start);
        body.push_str("call:tool_name{\"arg\":\"value\"}");
        body.push_str(&self.tokens.tool_call_end);
        Ok(body)
    }

    fn render_message(&self, message: &Gemma4Message) -> Result<String> {
        match message.role {
            Gemma4Role::System | Gemma4Role::User => {
                self.render_turn(message.role, &message.content)
            }
            Gemma4Role::Assistant => {
                let mut body = String::new();
                if let Some(thinking) = message.thinking.as_deref() {
                    body.push_str(&self.tokens.channel_start);
                    body.push_str("thought\n");
                    body.push_str(thinking);
                    body.push_str(&self.tokens.channel_end);
                }
                if !message.content.is_empty() {
                    if !body.is_empty() {
                        body.push('\n');
                    }
                    body.push_str(&message.content);
                }
                if !message.tool_calls.is_empty() {
                    for tool_call in &message.tool_calls {
                        if !body.is_empty() {
                            body.push('\n');
                        }
                        body.push_str(&self.tokens.tool_call_start);
                        body.push_str("call:");
                        body.push_str(&tool_call.name);
                        body.push_str(&serde_json::to_string(&tool_call.arguments)?);
                        body.push_str(&self.tokens.tool_call_end);
                    }
                }
                self.render_turn(Gemma4Role::Assistant, &body)
            }
            Gemma4Role::Tool => {
                let tool_name = message.tool_name.as_deref().ok_or_else(|| {
                    Error::Model("Gemma4 tool messages must include a tool_name".to_string())
                })?;
                let tool_content = parse_json_or_string(&message.content);
                let mut body = String::new();
                body.push_str(tool_name);
                body.push('\n');
                body.push_str(&self.tokens.tool_response_start);
                body.push_str(&serialize_tool_response(tool_name, tool_content)?);
                body.push_str(&self.tokens.tool_response_end);
                self.render_turn(Gemma4Role::Tool, &body)
            }
        }
    }

    fn render_turn(&self, role: Gemma4Role, body: &str) -> Result<String> {
        if body.contains(&self.tokens.turn_start) {
            return Err(Error::Model(format!(
                "Gemma4 turn body for role {} contains the turn start token",
                role.as_str()
            )));
        }

        Ok(format!(
            "{}{}\n{}\n{}\n",
            self.tokens.turn_start,
            role.as_str(),
            body,
            self.tokens.turn_end
        ))
    }

    pub fn has_complete_tool_call(&self, text: &str) -> bool {
        text.contains(&self.tokens.tool_call_start) && text.contains(&self.tokens.tool_call_end)
    }

    pub fn parse_assistant_response(
        &self,
        raw_text: &str,
    ) -> Result<Gemma4ParsedAssistantResponse> {
        let mut remaining = raw_text.to_string();
        let thinking = self.extract_thinking(&mut remaining)?;
        let tool_calls = self.extract_tool_calls(&mut remaining)?;
        let mut content = remaining.trim().to_string();
        loop {
            let trimmed = content.trim_end();
            if let Some(next) = trimmed.strip_suffix(&self.tokens.turn_end) {
                content = next.trim_end().to_string();
                continue;
            }
            if let Some(next) = trimmed.strip_suffix(&self.tokens.eos) {
                content = next.trim_end().to_string();
                continue;
            }
            content = trimmed.to_string();
            break;
        }

        Ok(Gemma4ParsedAssistantResponse {
            raw_text: raw_text.to_string(),
            content,
            thinking,
            tool_calls,
        })
    }

    fn extract_thinking(&self, remaining: &mut String) -> Result<Option<String>> {
        let start = format!("{}thought\n", self.tokens.channel_start);
        let Some(start_idx) = remaining.find(&start) else {
            return Ok(None);
        };
        let thinking_start = start_idx + start.len();
        let Some(end_rel) = remaining[thinking_start..].find(&self.tokens.channel_end) else {
            return Err(Error::Model(
                "Gemma4 assistant thinking block is missing the channel end token".to_string(),
            ));
        };
        let end_idx = thinking_start + end_rel;
        let thinking = remaining[thinking_start..end_idx].to_string();
        let mut remove_end = end_idx + self.tokens.channel_end.len();
        if remaining[remove_end..].starts_with('\n') {
            remove_end += 1;
        }
        remaining.replace_range(start_idx..remove_end, "");
        Ok(Some(thinking))
    }

    fn extract_tool_calls(&self, remaining: &mut String) -> Result<Vec<Gemma4ToolCall>> {
        let mut tool_calls = Vec::new();

        while let Some(start_idx) = remaining.find(&self.tokens.tool_call_start) {
            let payload_start = start_idx + self.tokens.tool_call_start.len();
            let Some(end_rel) = remaining[payload_start..].find(&self.tokens.tool_call_end) else {
                break;
            };
            let payload_end = payload_start + end_rel;
            let payload = remaining[payload_start..payload_end].trim();
            let tool_call = parse_tool_call_payload(payload)?;
            tool_calls.push(tool_call);

            let mut remove_end = payload_end + self.tokens.tool_call_end.len();
            if remaining[remove_end..].starts_with('\n') {
                remove_end += 1;
            }
            remaining.replace_range(start_idx..remove_end, "");
        }

        Ok(tool_calls)
    }
}

#[derive(Debug, Deserialize)]
struct TokenizerConfigTokens {
    #[serde(default)]
    bos_token: Option<String>,
    #[serde(default)]
    eos_token: Option<String>,
    #[serde(default)]
    sot_token: Option<String>,
    #[serde(default)]
    eot_token: Option<String>,
    #[serde(default)]
    soc_token: Option<String>,
    #[serde(default)]
    eoc_token: Option<String>,
    #[serde(default)]
    std_token: Option<String>,
    #[serde(default)]
    etd_token: Option<String>,
    #[serde(default)]
    stc_token: Option<String>,
    #[serde(default)]
    etc_token: Option<String>,
    #[serde(default)]
    str_token: Option<String>,
    #[serde(default)]
    etr_token: Option<String>,
}

fn parse_json_or_string(content: &str) -> Value {
    serde_json::from_str(content).unwrap_or_else(|_| Value::String(content.to_string()))
}

fn serialize_tool_spec(tool: &Gemma4ToolSpec) -> Result<String> {
    let mut map = Map::new();
    map.insert(
        "description".to_string(),
        Value::String(tool.description.clone()),
    );
    map.insert("name".to_string(), Value::String(tool.name.clone()));
    map.insert("parameters".to_string(), tool.parameters.clone());
    serde_json::to_string(&Value::Object(map)).map_err(Into::into)
}

fn serialize_tool_response(tool_name: &str, content: Value) -> Result<String> {
    let mut map = Map::new();
    map.insert("content".to_string(), content);
    map.insert("name".to_string(), Value::String(tool_name.to_string()));
    serde_json::to_string(&Value::Object(map)).map_err(Into::into)
}

fn parse_tool_call_payload(payload: &str) -> Result<Gemma4ToolCall> {
    if let Some(rest) = payload.strip_prefix("call:") {
        let brace_idx = rest.find('{').ok_or_else(|| {
            Error::Model("Gemma4 tool call payload is missing JSON arguments".to_string())
        })?;
        let name = rest[..brace_idx].trim();
        if name.is_empty() {
            return Err(Error::Model(
                "Gemma4 tool call payload is missing a tool name".to_string(),
            ));
        }
        let arguments = serde_json::from_str(&rest[brace_idx..])?;
        return Ok(Gemma4ToolCall::new(name, arguments));
    }

    let value: Value = serde_json::from_str(payload)?;
    parse_json_tool_call(&value)
}

fn parse_json_tool_call(value: &Value) -> Result<Gemma4ToolCall> {
    let obj = value.as_object().ok_or_else(|| {
        Error::Model("Gemma4 tool call payload must be a JSON object".to_string())
    })?;

    if let (Some(name), Some(arguments)) = (
        obj.get("name").and_then(Value::as_str),
        obj.get("arguments"),
    ) {
        return Ok(Gemma4ToolCall::new(name, arguments.clone()));
    }
    if let (Some(name), Some(arguments)) = (
        obj.get("name").and_then(Value::as_str),
        obj.get("parameters"),
    ) {
        return Ok(Gemma4ToolCall::new(name, arguments.clone()));
    }
    if let Some(function) = obj.get("function").and_then(Value::as_object) {
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                Error::Model("Gemma4 nested tool call is missing function.name".to_string())
            })?;
        let arguments = function.get("arguments").ok_or_else(|| {
            Error::Model("Gemma4 nested tool call is missing function.arguments".to_string())
        })?;
        let arguments = match arguments {
            Value::String(raw) => serde_json::from_str(raw)?,
            other => other.clone(),
        };
        return Ok(Gemma4ToolCall::new(name, arguments));
    }

    Err(Error::Model(
        "Gemma4 tool call JSON must contain name+arguments, name+parameters, or function.{name,arguments}".to_string(),
    ))
}

fn run_tool_loop_with_generator<F>(
    _template: &Gemma4ChatTemplate,
    conversation: &mut Gemma4Conversation,
    tools: &Gemma4ToolRegistry,
    config: &Gemma4ChatConfig,
    mut generate: F,
) -> Result<Gemma4ChatResponse>
where
    F: FnMut(&[Gemma4Message], &[Gemma4ToolSpec]) -> Result<GeneratedAssistantTurn>,
{
    let mut tokens_generated = 0;
    let mut all_tool_calls = Vec::new();
    let mut all_tool_results = Vec::new();
    let mut tool_iterations = 0usize;

    loop {
        let tool_specs = tools.specs();
        let generated = generate(&conversation.messages, &tool_specs)?;
        tokens_generated += generated.tokens_generated;

        let assistant_message = Gemma4Message {
            role: Gemma4Role::Assistant,
            content: generated.parsed.content.clone(),
            thinking: generated.parsed.thinking.clone(),
            tool_calls: generated.parsed.tool_calls.clone(),
            tool_name: None,
        };
        conversation.add_assistant_message(assistant_message);

        if generated.parsed.tool_calls.is_empty() {
            return Ok(Gemma4ChatResponse {
                raw_text: generated.parsed.raw_text,
                text: generated.parsed.content,
                thinking: generated.parsed.thinking,
                tool_calls: all_tool_calls,
                tool_results: all_tool_results,
                tokens_generated,
            });
        }

        if tool_iterations >= config.max_tool_iterations {
            return Err(Error::Model(
                "Gemma4 chat reached max_tool_iterations before producing a final assistant response"
                    .to_string(),
            ));
        }
        tool_iterations += 1;

        let round_calls = generated.parsed.tool_calls;
        let round_results: Vec<_> = round_calls
            .iter()
            .map(|call| tools.execute_call(call))
            .collect();
        for result in &round_results {
            conversation.add_tool_result(result);
        }
        all_tool_calls.extend(round_calls);
        all_tool_results.extend(round_results);
    }
}

pub struct Gemma4ChatPipeline {
    pub model: Model,
    pub tokenizer: Tokenizer,
    pub template: Gemma4ChatTemplate,
    pub config: Gemma4ChatConfig,
    pub tools: Gemma4ToolRegistry,
}

impl Gemma4ChatPipeline {
    pub fn load(model_dir: impl AsRef<Path>, config: Gemma4ChatConfig) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        Ok(Self {
            model: load_model(model_dir)?,
            tokenizer: load_tokenizer(model_dir)?,
            template: Gemma4ChatTemplate::load(model_dir)?,
            config,
            tools: Gemma4ToolRegistry::new(),
        })
    }

    pub fn register_tool<T>(&mut self, tool: T)
    where
        T: Gemma4Tool + 'static,
    {
        self.tools.register(tool);
    }

    pub fn chat(&mut self, conversation: &mut Gemma4Conversation) -> Result<Gemma4ChatResponse> {
        let template = self.template.clone();
        let config = self.config.clone();
        let tokenizer = self.tokenizer.clone();

        run_tool_loop_with_generator(
            &template,
            conversation,
            &self.tools,
            &config,
            |messages, tool_specs| {
                generate_assistant_turn(
                    &mut self.model,
                    &tokenizer,
                    &template,
                    messages,
                    tool_specs,
                    &config,
                )
            },
        )
    }
}

fn generate_assistant_turn(
    model: &mut Model,
    tokenizer: &Tokenizer,
    template: &Gemma4ChatTemplate,
    messages: &[Gemma4Message],
    tool_specs: &[Gemma4ToolSpec],
    config: &Gemma4ChatConfig,
) -> Result<GeneratedAssistantTurn> {
    let prompt = template.render_prompt(messages, tool_specs, true)?;
    let raw_text = generate_raw_text(model, tokenizer, &prompt, config, template)?;
    let parsed = template.parse_assistant_response(&raw_text.0)?;
    Ok(GeneratedAssistantTurn::new(parsed, raw_text.1))
}

/// EOS token IDs from Gemma4 generation_config.json.
/// Includes: eos (1), turn_end (106), pad (50).
pub const EOS_TOKEN_IDS: &[u32] = &[1, 106, 50];

fn generate_raw_text(
    model: &mut Model,
    tokenizer: &Tokenizer,
    prompt: &str,
    config: &Gemma4ChatConfig,
    _template: &Gemma4ChatTemplate,
) -> Result<(String, usize)> {
    let encoding = tokenizer.encode(prompt, false)?;
    let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);
    let mut cache = Vec::<KVCache>::new();
    let generator = Generate::new(model, &mut cache, config.temperature, &prompt_tokens);

    let mut generated_ids = Vec::new();
    for token in generator.take(config.max_new_tokens) {
        let token = token?;
        let token_id = token.item::<u32>();
        generated_ids.push(token_id);

        // Token-level stop: EOS (1), turn_end (106), pad (50)
        if EOS_TOKEN_IDS.contains(&token_id) {
            break;
        }
    }

    let decoded = tokenizer.decode(&generated_ids, false)?;
    Ok((decoded, generated_ids.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn render_prompt_uses_default_system_prompt_and_generation_prompt() {
        let template = Gemma4ChatTemplate::default();
        let messages = [Gemma4Message::user("Hello Gemma")];

        let rendered = template.render_prompt(&messages, &[], true).unwrap();

        assert_eq!(
            rendered,
            concat!(
                "<bos><|turn>system\n",
                "You are a helpful assistant.\n",
                "<turn|>\n",
                "<|turn>user\n",
                "Hello Gemma\n",
                "<turn|>\n",
                "<|turn>assistant\n",
            )
        );
    }

    #[test]
    fn render_prompt_includes_tools_assistant_tool_calls_and_tool_responses() {
        let template = Gemma4ChatTemplate::default();
        let tools = [Gemma4ToolSpec::new(
            "weather",
            "Lookup weather",
            json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"]
            }),
        )];
        let messages = [
            Gemma4Message::system("You are a weather assistant."),
            Gemma4Message::user("What is the weather in Paris?"),
            Gemma4Message::assistant_with_tool_calls(
                "Let me check.",
                vec![Gemma4ToolCall::new("weather", json!({ "city": "Paris" }))],
            ),
            Gemma4Message::tool(
                "weather",
                json!({"temp_c": 20, "condition": "sunny"}).to_string(),
            ),
        ];

        let rendered = template.render_prompt(&messages, &tools, true).unwrap();

        assert!(rendered.starts_with("<bos><|turn>system\nYou are a weather assistant.\n"));
        assert!(rendered.contains(
            "Available tools:\n<|tool>{\"description\":\"Lookup weather\",\"name\":\"weather\",\"parameters\":{\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"],\"type\":\"object\"}}<tool|>\n"
        ));
        assert!(rendered
            .contains("<|tool_call>call:weather{\"city\":\"Paris\"}<tool_call|>\n<turn|>\n"));
        assert!(rendered.contains(
            "<|turn>tool\nweather\n<|tool_response>{\"content\":{\"condition\":\"sunny\",\"temp_c\":20},\"name\":\"weather\"}<tool_response|>\n<turn|>\n"
        ));
        assert!(rendered.ends_with("<|turn>assistant\n"));
    }

    #[test]
    fn parse_assistant_response_extracts_thinking_content_and_tool_calls() {
        let template = Gemma4ChatTemplate::default();
        let raw = concat!(
            "<|channel>thought\n",
            "Need to look this up",
            "<channel|>\n",
            "Let me check.\n",
            "<|tool_call>call:weather{\"city\":\"Paris\"}<tool_call|>",
            "<turn|>"
        );

        let parsed = template.parse_assistant_response(raw).unwrap();

        assert_eq!(parsed.thinking.as_deref(), Some("Need to look this up"));
        assert_eq!(parsed.content, "Let me check.");
        assert_eq!(
            parsed.tool_calls,
            vec![Gemma4ToolCall::new("weather", json!({ "city": "Paris" }))]
        );
    }

    #[test]
    fn parse_assistant_response_accepts_json_tool_call_payloads() {
        let template = Gemma4ChatTemplate::default();
        let raw = concat!(
            "<|tool_call>",
            "{\"name\":\"weather\",\"arguments\":{\"city\":\"Paris\"}}",
            "<tool_call|><turn|>"
        );

        let parsed = template.parse_assistant_response(raw).unwrap();

        assert_eq!(parsed.content, "");
        assert_eq!(
            parsed.tool_calls,
            vec![Gemma4ToolCall::new("weather", json!({ "city": "Paris" }))]
        );
    }

    #[test]
    fn has_complete_tool_call_requires_end_marker() {
        let template = Gemma4ChatTemplate::default();

        assert!(template
            .has_complete_tool_call("<|tool_call>call:weather{\"city\":\"Paris\"}<tool_call|>"));
        assert!(!template.has_complete_tool_call("<|tool_call>call:weather{\"city\":\"Paris\"}"));
    }

    #[test]
    fn parse_assistant_response_strips_trailing_eos_token() {
        let template = Gemma4ChatTemplate::default();

        let parsed = template.parse_assistant_response("Done.<eos>").unwrap();

        assert_eq!(parsed.content, "Done.");
    }

    #[test]
    fn tool_loop_executes_tools_and_continues_to_final_answer() {
        let template = Gemma4ChatTemplate::default();
        let mut conversation = Gemma4Conversation::new();
        conversation.add_user("What is the weather in Paris?");

        let mut tools = Gemma4ToolRegistry::new();
        tools.register(Gemma4FunctionTool::new(
            Gemma4ToolSpec::new(
                "weather",
                "Lookup weather",
                json!({
                    "type": "object",
                    "properties": {
                        "city": { "type": "string" }
                    },
                    "required": ["city"]
                }),
            ),
            |arguments| {
                Ok(json!({
                    "city": arguments["city"].clone(),
                    "temp_c": 20,
                    "condition": "sunny"
                }))
            },
        ));

        let config = Gemma4ChatConfig {
            max_tool_iterations: 2,
            ..Default::default()
        };
        let mut generation_count = 0;

        let response = run_tool_loop_with_generator(
            &template,
            &mut conversation,
            &tools,
            &config,
            |messages, tool_specs| {
                generation_count += 1;
                match generation_count {
                    1 => {
                        assert_eq!(tool_specs.len(), 1);
                        assert!(messages.iter().any(|message| message.role == Gemma4Role::User));
                        Ok(GeneratedAssistantTurn::new(
                            Gemma4ParsedAssistantResponse {
                                raw_text: "<|tool_call>call:weather{\"city\":\"Paris\"}<tool_call|><turn|>"
                                    .to_string(),
                                content: String::new(),
                                thinking: Some("Need weather data".to_string()),
                                tool_calls: vec![Gemma4ToolCall::new(
                                    "weather",
                                    json!({ "city": "Paris" }),
                                )],
                            },
                            7,
                        ))
                    }
                    2 => {
                        assert!(messages.iter().any(|message| {
                            message.role == Gemma4Role::Tool
                                && message.content.contains("\"temp_c\":20")
                        }));
                        Ok(GeneratedAssistantTurn::new(
                            Gemma4ParsedAssistantResponse {
                                raw_text: "It is 20C and sunny.<turn|>".to_string(),
                                content: "It is 20C and sunny.".to_string(),
                                thinking: None,
                                tool_calls: Vec::new(),
                            },
                            5,
                        ))
                    }
                    _ => panic!("unexpected generation count"),
                }
            },
        )
        .unwrap();

        assert_eq!(response.text, "It is 20C and sunny.");
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_results.len(), 1);
        assert_eq!(response.tool_results[0].name, "weather");
        assert_eq!(response.tool_results[0].content["temp_c"], 20);
        assert_eq!(response.tokens_generated, 12);
    }

    #[test]
    fn tool_loop_returns_plain_assistant_response_without_tools() {
        let template = Gemma4ChatTemplate::default();
        let mut conversation = Gemma4Conversation::new();
        conversation.add_user("Say hello");
        let tools = Gemma4ToolRegistry::new();
        let config = Gemma4ChatConfig::default();

        let response = run_tool_loop_with_generator(
            &template,
            &mut conversation,
            &tools,
            &config,
            |messages, tool_specs| {
                assert_eq!(messages.len(), 1);
                assert!(tool_specs.is_empty());
                Ok(GeneratedAssistantTurn::new(
                    Gemma4ParsedAssistantResponse {
                        raw_text: "Hello there!<turn|>".to_string(),
                        content: "Hello there!".to_string(),
                        thinking: None,
                        tool_calls: Vec::new(),
                    },
                    3,
                ))
            },
        )
        .unwrap();

        assert_eq!(response.text, "Hello there!");
        assert!(response.tool_calls.is_empty());
        assert!(response.tool_results.is_empty());
        assert_eq!(conversation.messages.len(), 2);
        assert_eq!(conversation.messages[1].role, Gemma4Role::Assistant);
    }

    #[test]
    fn tool_loop_errors_after_max_tool_iterations() {
        let template = Gemma4ChatTemplate::default();
        let mut conversation = Gemma4Conversation::new();
        conversation.add_user("Keep calling the tool");

        let mut tools = Gemma4ToolRegistry::new();
        tools.register(Gemma4FunctionTool::new(
            Gemma4ToolSpec::new("weather", "Lookup weather", json!({"type": "object"})),
            |_| Ok(json!({"ok": true})),
        ));

        let config = Gemma4ChatConfig {
            max_tool_iterations: 1,
            ..Default::default()
        };
        let mut generation_count = 0;

        let error = run_tool_loop_with_generator(
            &template,
            &mut conversation,
            &tools,
            &config,
            |_messages, _tool_specs| {
                generation_count += 1;
                Ok(GeneratedAssistantTurn::new(
                    Gemma4ParsedAssistantResponse {
                        raw_text: "<|tool_call>call:weather{\"city\":\"Paris\"}<tool_call|><turn|>"
                            .to_string(),
                        content: String::new(),
                        thinking: None,
                        tool_calls: vec![Gemma4ToolCall::new(
                            "weather",
                            json!({ "city": "Paris" }),
                        )],
                    },
                    2,
                ))
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("max_tool_iterations"));
        assert_eq!(generation_count, 2);
    }
}
