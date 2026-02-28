use mlx_rs::{
    error::Exception,
    macros::ModuleParameters,
    module::Module,
    nn,
    quantization::MaybeQuantized,
};

/// SiLU-gated MLP (same as standard Qwen/LLaMA MLP).
#[derive(Debug, ModuleParameters)]
pub struct Mlp {
    #[param]
    pub gate_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub down_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub up_proj: MaybeQuantized<nn::Linear>,
}

impl Module<&mlx_rs::Array> for Mlp {
    type Output = mlx_rs::Array;
    type Error = Exception;

    fn forward(&mut self, x: &mlx_rs::Array) -> Result<Self::Output, Self::Error> {
        let gate = nn::silu(self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&gate.multiply(up)?)
    }

    fn training_mode(&mut self, _mode: bool) {}
}
