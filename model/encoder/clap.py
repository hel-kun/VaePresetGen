import torch
import torch.nn as nn
from transformers import ClapTextModelWithProjection, AutoProcessor, AutoTokenizer, AutoModel, ClapAudioModelWithProjection

class CLAPTextEncorder(nn.Module):
    def __init__(
        self,
        model_name="laion/clap-htsat-unfused",
        output_dim=512,
    ):
        super().__init__()
        self.model = ClapTextModelWithProjection.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        if self.model.config.projection_dim != output_dim:
            self.project = nn.Linear(self.model.config.projection_dim, output_dim)
        else:
            self.project = nn.Identity()

        self.mu_head = nn.Linear(output_dim, output_dim)
        self.logvar_head = nn.Linear(output_dim, output_dim)
    
    def forward(self, text, enable_grad: bool = False):
        # テキストの前処理
        inputs = self.processor(
            text=text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
        
        ctx = torch.enable_grad() if enable_grad else torch.no_grad()
        with ctx:
            outputs = self.model(**inputs)
        
        embeddings = outputs.text_embeds  # (batch_size, projection_dim)
        embeddings = self.project(embeddings) # 次元変換
        mu = self.mu_head(embeddings)
        logvar = self.logvar_head(embeddings)
        return mu, logvar

class CLAPAudioEncorder(nn.Module):
    def __init__(
        self,
        model_name="laion/clap-htsat-unfused",
        output_dim=512,
    ):
        super().__init__()
        self.model = ClapAudioModelWithProjection.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        if self.model.config.projection_dim != output_dim:
            self.project = nn.Linear(self.model.config.projection_dim, output_dim)
        else:
            self.project = nn.Identity()
        
        self.mu_head = nn.Linear(output_dim, output_dim)
        self.logvar_head = nn.Linear(output_dim, output_dim)
    
    def forward(self, audio):
        # 音声の前処理
        inputs = self.processor(
            audio=audio,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embeddings = outputs.audio_embeds  # (batch_size, projection_dim)
        embeddings = self.project(embeddings) # 次元変換
        mu = self.mu_head(embeddings)
        logvar = self.logvar_head(embeddings)
        return mu, logvar