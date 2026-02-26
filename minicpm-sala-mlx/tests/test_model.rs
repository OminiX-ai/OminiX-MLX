use minicpm_sala_mlx::config::ModelArgs;

#[test]
fn test_config_parsing() {
    let json = r#"{
        "hidden_size": 4096,
        "intermediate_size": 16384,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 2,
        "vocab_size": 73448,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "scale_emb": 12,
        "scale_depth": 1.4,
        "dim_model_base": 256,
        "attn_use_rope": false,
        "lightning_use_rope": true,
        "qk_norm": true,
        "use_output_gate": true,
        "use_output_norm": true,
        "attn_use_output_gate": true,
        "lightning_nh": 32,
        "lightning_nkv": 32,
        "lightning_head_dim": 128,
        "lightning_scale": "1/sqrt(d)",
        "mixer_types": [
            "minicpm4", "lightning-attn", "lightning-attn", "lightning-attn",
            "lightning-attn", "lightning-attn", "lightning-attn", "lightning-attn",
            "lightning-attn", "minicpm4", "lightning-attn", "lightning-attn",
            "lightning-attn", "lightning-attn", "lightning-attn", "lightning-attn",
            "minicpm4", "minicpm4", "lightning-attn", "lightning-attn",
            "lightning-attn", "lightning-attn", "minicpm4", "lightning-attn",
            "lightning-attn", "lightning-attn", "lightning-attn", "lightning-attn",
            "lightning-attn", "minicpm4", "minicpm4", "minicpm4"
        ],
        "sparse_config": {
            "kernel_size": 32,
            "kernel_stride": 16,
            "init_blocks": 1,
            "block_size": 64,
            "window_size": 2048,
            "topk": 64,
            "use_nope": false,
            "dense_len": 8192
        }
    }"#;

    let args: ModelArgs = serde_json::from_str(json).unwrap();

    assert_eq!(args.hidden_size, 4096);
    assert_eq!(args.num_hidden_layers, 32);
    assert_eq!(args.vocab_size, 73448);
    assert_eq!(args.num_key_value_heads, 2);
    assert_eq!(args.scale_emb, 12.0);
    assert_eq!(args.scale_depth, 1.4);
    assert!(!args.attn_use_rope);
    assert!(args.lightning_use_rope);
    assert!(args.qk_norm);
    assert!(args.use_output_gate);
    assert!(args.attn_use_output_gate);

    // Layer type checks — sparse at: 0, 9, 16, 17, 22, 29, 30, 31
    assert!(args.is_sparse_layer(0));
    assert!(!args.is_sparse_layer(1));
    assert!(args.is_sparse_layer(9));
    assert!(!args.is_sparse_layer(15));
    assert!(args.is_sparse_layer(16));
    assert!(args.is_sparse_layer(17));
    assert!(args.is_sparse_layer(22));
    assert!(args.is_sparse_layer(31));
    assert_eq!(
        args.mixer_types.iter().filter(|t| *t == "minicpm4").count(),
        8
    );

    // Derived values
    assert_eq!(args.lightning_num_heads(), 32);
    assert_eq!(args.lightning_num_kv_heads(), 32);
    assert!((args.residual_scale() - 1.4 / (32.0_f32).sqrt()).abs() < 1e-6);
    assert!((args.logits_scale() - 16.0).abs() < 1e-6);
    assert!((args.lightning_scale_value() - (128.0_f32).sqrt().recip()).abs() < 1e-6);

    // Sparse config
    let sc = args.sparse_config.as_ref().unwrap();
    assert_eq!(sc.block_size, 64);
    assert_eq!(sc.topk, 64);
    assert_eq!(sc.dense_len, 8192);
}
