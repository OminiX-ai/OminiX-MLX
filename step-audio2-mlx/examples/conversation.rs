//! Multi-turn Conversation Example
//!
//! Demonstrates Step-Audio 2's conversation capabilities with:
//! - Multi-turn dialogue
//! - Think mode
//! - Tool calling
//!
//! Usage:
//!     cargo run --example conversation -- <model_path>
//!
//! Example:
//!     cargo run --example conversation -- ./Step-Audio-2-mini

use std::env;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

use step_audio2_mlx::{
    StepAudio2Pipeline, PipelineConfig, SamplingConfig,
    Result, Error,
};
use step_audio2_mlx::pipeline::{Conversation, ConversationTurn, load_audio};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Step-Audio 2 Conversation Example");
        eprintln!("");
        eprintln!("Usage: {} <model_path> [--think] [--tools]", args[0]);
        eprintln!("");
        eprintln!("Arguments:");
        eprintln!("  model_path  Path to Step-Audio-2-mini model directory");
        eprintln!("");
        eprintln!("Options:");
        eprintln!("  --think     Enable think mode");
        eprintln!("  --tools     Enable tool calling");
        eprintln!("");
        eprintln!("Example:");
        eprintln!("  {} ./Step-Audio-2-mini --think", args[0]);
        return Err(Error::Config("Invalid arguments".into()));
    }

    let model_path = PathBuf::from(&args[1]);
    let enable_think = args.iter().any(|a| a == "--think");
    let enable_tools = args.iter().any(|a| a == "--tools");

    // Validate path
    if !model_path.exists() {
        return Err(Error::ModelLoad(format!(
            "Model path not found: {}",
            model_path.display()
        )));
    }

    println!("Step-Audio 2 Conversation");
    println!("=========================");
    println!("");
    println!("Model: {}", model_path.display());
    println!("Think mode: {}", if enable_think { "enabled" } else { "disabled" });
    println!("Tool calling: {}", if enable_tools { "enabled" } else { "disabled" });
    println!("");

    // Configure pipeline
    let config = PipelineConfig::default()
        .think(enable_think)
        .tools(enable_tools)
        .with_sampling(SamplingConfig::balanced());

    // Load pipeline
    println!("Loading model...");
    let start = Instant::now();
    let mut pipeline = StepAudio2Pipeline::load(&model_path, config)?;
    println!("Model loaded in {:.2}s", start.elapsed().as_secs_f64());
    println!("");

    // Set system prompt
    pipeline.set_system_prompt("You are a helpful AI assistant.");

    // Show available commands
    println!("Commands:");
    println!("  /audio <path>  - Process audio file");
    println!("  /clear         - Clear conversation history");
    println!("  /quit          - Exit");
    println!("");

    // Interactive loop
    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        // Handle commands
        if input.starts_with('/') {
            let parts: Vec<&str> = input.splitn(2, ' ').collect();
            let command = parts[0];

            match command {
                "/quit" | "/exit" | "/q" => {
                    println!("Goodbye!");
                    break;
                }
                "/clear" => {
                    pipeline.clear_conversation();
                    println!("Conversation cleared.");
                    continue;
                }
                "/audio" => {
                    if parts.len() < 2 {
                        println!("Usage: /audio <path>");
                        continue;
                    }
                    let audio_path = PathBuf::from(parts[1]);
                    if !audio_path.exists() {
                        println!("Audio file not found: {}", audio_path.display());
                        continue;
                    }

                    // Load and process audio
                    match load_audio(&audio_path) {
                        Ok((samples, sample_rate)) => {
                            println!("Processing audio...");
                            let start = Instant::now();

                            match pipeline.chat_audio(&samples, sample_rate) {
                                Ok(response) => {
                                    let duration = start.elapsed();

                                    if let Some(thinking) = &response.thinking {
                                        println!("");
                                        println!("Thinking: {}", thinking);
                                    }

                                    println!("");
                                    println!("Assistant: {}", response.text);

                                    if !response.tool_calls.is_empty() {
                                        println!("");
                                        println!("Tool calls: {}", response.tool_calls.len());
                                    }

                                    println!("");
                                    println!(
                                        "[{} tokens in {:.2}s]",
                                        response.tokens_generated,
                                        duration.as_secs_f64()
                                    );
                                }
                                Err(e) => {
                                    println!("Error: {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            println!("Error loading audio: {}", e);
                        }
                    }
                    continue;
                }
                _ => {
                    println!("Unknown command: {}", command);
                    continue;
                }
            }
        }

        // Text input (currently not supported)
        println!("Text input not yet supported. Use /audio <path> to process audio files.");
    }

    Ok(())
}
