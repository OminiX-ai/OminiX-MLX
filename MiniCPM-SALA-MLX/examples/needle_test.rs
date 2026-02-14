use std::time::Instant;

use clap::Parser;
use minicpm_sala_mlx::{
    create_layer_caches, is_stop_token, load_model, load_tokenizer, sample, strip_thinking,
};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::transforms::eval;

#[derive(Parser)]
#[command(name = "needle_test", about = "Needle-in-a-haystack long context test")]
struct Args {
    /// Path to model directory
    model_dir: String,

    /// Target context length in tokens (e.g. 32000, 128000, 512000)
    #[arg(long, default_value_t = 32000)]
    context_len: usize,

    /// Needle insertion depth as fraction (0.0 = start, 1.0 = end)
    #[arg(long, default_value_t = 0.25)]
    depth: f32,

    /// Maximum tokens to generate for the answer
    #[arg(long, default_value_t = 64)]
    max_tokens: usize,
}

const NEEDLE: &str =
    "The secret verification code for Project Aurora is 739258. Remember this number carefully.";

const FILLER_PARAGRAPHS: &[&str] = &[
    "The annual rainfall in the Pacific Northwest varies between 150 and 200 centimeters depending on elevation and proximity to the coast. Meteorologists track these patterns using a network of ground stations and satellite imagery. The data collected helps farmers plan their crop rotations and irrigation schedules throughout the growing season.",
    "Modern bread baking techniques combine traditional fermentation methods with precise temperature control. The ideal proofing temperature for sourdough ranges from 24 to 27 degrees Celsius. Professional bakers monitor dough hydration levels carefully, as even small variations can significantly affect the final crumb structure and crust development.",
    "The history of lighthouse construction along the Atlantic seaboard spans three centuries of engineering innovation. Early wooden structures gave way to stone towers in the 1800s. Fresnel lens technology revolutionized the range of lighthouse beams, allowing ships to navigate dangerous coastlines from distances exceeding twenty nautical miles.",
    "Urban planning in European cities has evolved to prioritize pedestrian access and public transportation. Cities like Amsterdam and Copenhagen have invested heavily in bicycle infrastructure. Studies show that reducing car dependency improves air quality metrics by 15 to 30 percent while increasing retail activity in city centers.",
    "The migration patterns of monarch butterflies remain one of the most remarkable phenomena in the insect world. Each autumn, millions travel up to 4,800 kilometers from Canada to central Mexico. Researchers use tiny radio transmitters and isotope analysis to track individual butterflies across this incredible journey.",
    "Volcanic soil in regions like the Azores and Hawaii produces exceptionally fertile farmland. The high mineral content supports diverse agricultural output including coffee, pineapple, and various tropical fruits. Farmers on volcanic islands often maintain terraced fields that follow the natural contours of ancient lava flows.",
    "The development of fiber optic cables in the 1970s transformed global telecommunications. A single modern fiber can carry over 100 terabits per second across ocean floors. Submarine cable networks now span more than 1.3 million kilometers, forming the backbone of international internet connectivity.",
    "Traditional pottery techniques in East Asia involve multiple firing stages at temperatures exceeding 1200 degrees Celsius. Celadon glazes achieve their distinctive green hue through iron oxide reduction in oxygen-depleted kilns. Master potters spend decades perfecting the subtle variations that distinguish premium ceramics from ordinary ware.",
    "Antarctic research stations operate in extreme conditions with winter temperatures dropping below minus 60 degrees Celsius. Scientists stationed there study ice cores that contain atmospheric records spanning 800,000 years. These frozen archives provide invaluable data about historical climate patterns and greenhouse gas concentrations.",
    "The acoustic properties of concert halls depend on complex interactions between room geometry, surface materials, and air temperature. Engineers use computational fluid dynamics to model sound wave propagation. The reverberation time, measured in seconds, determines whether a hall is suited for orchestral music, chamber ensembles, or spoken word.",
    "Coral reef ecosystems support approximately 25 percent of all marine species despite covering less than 1 percent of the ocean floor. Reef-building corals depend on symbiotic algae called zooxanthellae for their energy. Rising ocean temperatures cause coral bleaching events that threaten biodiversity in tropical marine environments.",
    "The standardization of railroad gauge width in the 19th century was one of the most consequential infrastructure decisions in history. The 1435mm standard gauge, originally used by George Stephenson, became dominant across Europe and North America. Countries that adopted different gauges faced significant logistical challenges in cross-border freight transport.",
    "Archaeological discoveries at Gobekli Tepe in southeastern Turkey have fundamentally altered our understanding of Neolithic societies. The site features massive stone pillars carved with animal reliefs dating to approximately 9500 BCE. This predates agriculture and settled life, suggesting that monumental construction may have driven the transition to farming rather than the reverse.",
    "The chemistry of fermentation involves complex enzymatic pathways that convert sugars into ethanol and carbon dioxide. Saccharomyces cerevisiae, common baker's yeast, has been used for millennia in bread and beverage production. Modern genomic analysis has identified over 1,500 yeast strains with distinct fermentation characteristics used across different culinary traditions.",
    "Satellite imagery analysis reveals that global forest cover has declined by approximately 4.7 million hectares annually over the past two decades. Secondary growth forests, while valuable for carbon sequestration, support significantly less biodiversity than primary forests. Reforestation efforts must carefully consider native species composition to maximize ecological benefit.",
    "The physics of bridge design requires balancing tensile and compressive forces across span lengths ranging from meters to kilometers. Suspension bridges use high-strength steel cables that can support loads exceeding 200,000 tonnes. Modern computational modeling allows engineers to simulate wind load, seismic activity, and thermal expansion with unprecedented accuracy.",
    "Deep sea hydrothermal vents support unique ecosystems powered entirely by chemosynthesis rather than photosynthesis. Tube worms, clams, and specialized bacteria thrive at temperatures and pressures that would be lethal to most organisms. These extreme environments provide insights into the potential for life on other planets and moons in our solar system.",
    "The printing press, developed by Johannes Gutenberg around 1440, dramatically reduced the cost of book production. Before movable type, a single manuscript could take months to copy by hand. Within fifty years of its introduction, an estimated 20 million volumes had been printed across Europe, fundamentally transforming the spread of knowledge.",
    "Alpine glaciers serve as critical freshwater reservoirs for hundreds of millions of people downstream. The Rhone, Rhine, and Danube rivers all depend on glacial meltwater during summer months. Current retreat rates suggest that many smaller glaciers may disappear entirely within the next century, creating significant water security challenges.",
    "The human gut microbiome contains approximately 100 trillion microorganisms representing over 1,000 distinct species. Research has linked gut bacteria composition to conditions ranging from obesity to depression. Dietary fiber serves as a primary fuel source for beneficial microbes, producing short-chain fatty acids that support intestinal health.",
];

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    eprintln!("=== Needle-in-a-Haystack Test ===");
    eprintln!("Target context: {} tokens", args.context_len);
    eprintln!("Needle depth:   {:.0}%", args.depth * 100.0);
    eprintln!();

    // Load model and tokenizer
    let load_start = Instant::now();
    eprintln!("Loading model from {}...", args.model_dir);
    let tokenizer = load_tokenizer(&args.model_dir)?;
    let mut model = load_model(&args.model_dir)?;
    eprintln!("Model loaded in {:.1}s", load_start.elapsed().as_secs_f32());

    // Build filler text until we reach target token count.
    // We construct a long document, insert the needle at the specified depth,
    // then append a question at the end.
    let question = "Based on everything you have read above, what is the secret verification code for Project Aurora? Answer with just the number.";

    // Estimate tokens: tokenize the question + chat framing overhead
    let chat_prefix = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n";
    let chat_suffix = format!("\n\n{question}<|im_end|>\n<|im_start|>assistant\n");

    let prefix_tokens = tokenizer
        .encode(chat_prefix, false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .len();
    let suffix_tokens = tokenizer
        .encode(chat_suffix.as_str(), false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .len();
    let needle_tokens = tokenizer
        .encode(NEEDLE, false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .len();

    // Filler budget
    let filler_budget = args
        .context_len
        .saturating_sub(prefix_tokens + suffix_tokens + needle_tokens + 10);

    eprintln!(
        "Building context: {} prefix + {} filler + {} needle + {} suffix tokens",
        prefix_tokens, filler_budget, needle_tokens, suffix_tokens
    );

    // Pre-tokenize filler paragraphs to know their token counts
    let mut para_tokens: Vec<usize> = Vec::new();
    for &para in FILLER_PARAGRAPHS {
        let enc = tokenizer
            .encode(para, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        para_tokens.push(enc.get_ids().len());
    }

    // Calculate needle insertion point in filler tokens
    let needle_pos_tokens = (filler_budget as f32 * args.depth) as usize;

    // Build filler with needle inserted
    let mut filler_parts: Vec<String> = Vec::new();
    let mut token_count = 0;
    let mut needle_inserted = false;
    let mut para_idx = 0;

    while token_count < filler_budget {
        let p = para_idx % FILLER_PARAGRAPHS.len();
        let pt = para_tokens[p];

        // Insert needle at the right position
        if !needle_inserted && token_count + pt >= needle_pos_tokens {
            filler_parts.push(format!("\n\n{NEEDLE}\n\n"));
            needle_inserted = true;
            token_count += needle_tokens + 4; // rough newline overhead
        }

        if token_count + pt > filler_budget {
            break;
        }

        filler_parts.push(format!("\n\n{}", FILLER_PARAGRAPHS[p]));
        token_count += pt + 2;
        para_idx += 1;
    }

    // If needle wasn't inserted yet (very short context), append it
    if !needle_inserted {
        filler_parts.push(format!("\n\n{NEEDLE}\n\n"));
    }

    let filler_text = filler_parts.join("");

    // Assemble full prompt
    let full_prompt = format!("{chat_prefix}{filler_text}\n\n{question}{}", "<|im_end|>\n<|im_start|>assistant\n");

    // Tokenize
    let encoding = tokenizer
        .encode(full_prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let prompt_tokens = encoding.get_ids();
    let prompt_len = prompt_tokens.len();

    eprintln!("Actual prompt length: {} tokens", prompt_len);
    eprintln!();

    // Create caches and run
    let mut caches = create_layer_caches(&model.args);

    let input = mlx_rs::Array::from_slice(
        &prompt_tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(),
        &[1, prompt_len as i32],
    );

    // Prefill
    eprintln!("Prefilling {} tokens...", prompt_len);
    let prefill_start = Instant::now();
    let logits = model.forward(&input, &mut caches)?;
    let last_logits = logits.index((.., -1, ..));
    let mut token = sample(&last_logits, 0.0)?; // greedy
    eval([&token])?;
    let prefill_time = prefill_start.elapsed().as_secs_f32();
    eprintln!(
        "Prefill: {:.1}s ({:.1} tok/s)",
        prefill_time,
        prompt_len as f32 / prefill_time
    );

    // Decode
    let mut generated_ids: Vec<u32> = Vec::new();
    let decode_start = Instant::now();

    for _ in 0..args.max_tokens {
        let token_id = token.item::<u32>();
        if is_stop_token(token_id) {
            break;
        }
        generated_ids.push(token_id);

        let input = token.reshape(&[1, 1])?;
        let logits = model.forward(&input, &mut caches)?;
        let last_logits = logits.index((.., -1, ..));
        token = sample(&last_logits, 0.0)?;
    }
    eval([&token])?;
    let decode_time = decode_start.elapsed().as_secs_f32();

    let answer = tokenizer
        .decode(&generated_ids, true)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let answer = strip_thinking(&answer);

    eprintln!(
        "Decode:  {:.1}s ({:.1} tok/s, {} tokens)",
        decode_time,
        generated_ids.len() as f32 / decode_time,
        generated_ids.len()
    );
    eprintln!();

    // Check result
    let found = answer.contains("739258");
    eprintln!("=== Result ===");
    eprintln!("Context length: {} tokens", prompt_len);
    eprintln!("Needle depth:   {:.0}%", args.depth * 100.0);
    eprintln!("Model answer:   {}", answer.trim());
    eprintln!(
        "Needle found:   {}",
        if found { "YES" } else { "NO" }
    );
    eprintln!(
        "Prefill speed:  {:.1} tok/s",
        prompt_len as f32 / prefill_time
    );
    eprintln!(
        "Decode speed:   {:.1} tok/s",
        generated_ids.len() as f32 / decode_time
    );

    Ok(())
}
