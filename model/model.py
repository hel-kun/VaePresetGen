import torch
import torch.nn as nn
import torch.nn.functional as F
from model.encoder.clap import CLAPTextEncorder, CLAPAudioEncorder
from model.decoder.dsa_transformer import PresetGenDecoder


class VaePresetGenModel(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        latent_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        self.text_encoder = CLAPTextEncorder(embed_dim=embed_dim, latent_dim=latent_dim)
        self.audio_encoder = CLAPAudioEncorder(embed_dim=embed_dim, latent_dim=latent_dim)
        self.latent_to_dec = nn.Linear(latent_dim, embed_dim)
        self.decoder = PresetGenDecoder(embed_dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, text_inputs, audio_inputs=None, mode: str = "text_only"):
        assert mode in ["audio_text", "text_only"]
    
        # prior: テキストから
        mu_txt, logvar_txt = self.text_encoder(text_inputs, enable_grad=(mode == "audio_text"))
    
        posterior = None
        if mode == "audio_text":
            if audio_inputs is None:
                raise ValueError("audio_text mode requires audio_inputs")
            mu_aud, logvar_aud = self.audio_encoder(audio_inputs)
            z = self.reparameterize(mu_aud, logvar_aud)
            posterior = {"mu": mu_aud, "logvar": logvar_aud}
        else:
            z = self.reparameterize(mu_txt, logvar_txt)
    
        memory = self.latent_to_dec(z).unsqueeze(1)
        decoder_outputs = self.decoder(memory=memory)
    
        return {
            "decoder_outputs": decoder_outputs,
            "prior": {"mu": mu_txt, "logvar": logvar_txt},
            "posterior": posterior,
            "latent": z,
        }