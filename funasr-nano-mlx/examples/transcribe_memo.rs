use funasr_nano_mlx::model::FunASRNano;

fn main() {
    let model_dir = "./Fun-ASR-Nano-2512";
    let audio_path = "/tmp/voice_memo.wav";

    println!("Loading model...");
    let mut model = FunASRNano::load(model_dir).expect("Failed to load model");

    println!("Transcribing...");
    let result = model.transcribe(audio_path).expect("Failed to transcribe");
    println!("\nResult: {}", result);
}
