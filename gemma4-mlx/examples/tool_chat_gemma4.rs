use std::{
    env,
    error::Error,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use gemma4_mlx::{
    Gemma4ChatConfig, Gemma4ChatPipeline, Gemma4Conversation, Gemma4FunctionTool, Gemma4Message,
    Gemma4ToolSpec,
};
use serde_json::json;

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut enable_tools = false;
    let mut positional = Vec::new();
    for arg in args {
        if arg == "--tools" {
            enable_tools = true;
        } else {
            positional.push(arg);
        }
    }

    let model_dir = positional
        .first()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/gemma-4-26B-A4B-it"));
    let prompt = positional.get(1).cloned().unwrap_or_else(|| {
        if enable_tools {
            "What time is it right now? Use a tool if helpful.".to_string()
        } else {
            "Write a short haiku about Gemma 4.".to_string()
        }
    });

    let mut pipeline = Gemma4ChatPipeline::load(&model_dir, Gemma4ChatConfig::default())?;
    if enable_tools {
        pipeline.register_tool(Gemma4FunctionTool::new(
            Gemma4ToolSpec::new(
                "get_unix_time",
                "Return the current Unix timestamp in seconds.",
                json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            ),
            |_| {
                let seconds = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_err(|e| gemma4_mlx::Error::Model(e.to_string()))?
                    .as_secs();
                Ok(json!({ "unix_time": seconds }))
            },
        ));
        pipeline.register_tool(Gemma4FunctionTool::new(
            Gemma4ToolSpec::new(
                "echo_text",
                "Echo text back to the model.",
                json!({
                    "type": "object",
                    "properties": {
                        "text": { "type": "string" }
                    },
                    "required": ["text"]
                }),
            ),
            |arguments| Ok(json!({ "text": arguments["text"].clone() })),
        ));
    }

    let mut conversation = Gemma4Conversation::new();
    if enable_tools {
        conversation.add_assistant_message(Gemma4Message::system(
            "You are a helpful assistant. Use tools when they help answer the user's request.",
        ));
    }
    conversation.add_user(prompt);

    let response = pipeline.chat(&mut conversation)?;

    if let Some(thinking) = response.thinking.as_deref() {
        eprintln!("thinking:\n{}\n", thinking);
    }
    if !response.tool_calls.is_empty() {
        eprintln!("tool calls: {}", response.tool_calls.len());
        for call in &response.tool_calls {
            eprintln!(
                "- {} {}",
                call.name,
                serde_json::to_string(&call.arguments)?
            );
        }
    }
    if !response.tool_results.is_empty() {
        eprintln!("tool results: {}", response.tool_results.len());
        for result in &response.tool_results {
            eprintln!(
                "- {} {}",
                result.name,
                serde_json::to_string(&result.content)?
            );
        }
    }

    println!("{}", response.text);

    Ok(())
}
