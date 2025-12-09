import argparse
import torch
from datetime import datetime
import logging
from model.model import VaePresetGenModel
from model.trainer import Trainer
from dataset.dataset import Synth1Dataset
import os
from config import DEVICE

def main(args):
    mode = "audio_text" if args.is_audio_encoder else "text_only"
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.info(f"mode={mode} kl_weight={args.kl_weight} param_loss_weight={args.param_loss_weight}")
    logger.info(f"embedding_dim={args.embedding_dim} heads={args.num_heads} layers={args.num_layers}")

    dataset = Synth1Dataset(logger=logger, embed_dim=args.embedding_dim)
    model = VaePresetGenModel(
        embed_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    trainer = Trainer(
        model=model,
        dataset=dataset,
        optimizer=torch.optim.Adam(model.parameters(), lr=args.learning_rate),
        scheduler=None,
        log_interval=args.log_interval,
        early_stopping_patience=args.es_patience,
        logger=logger,
        kl_weight=args.kl_weight,
        kl_warmup_steps=args.kl_warmup_steps,
        param_loss_weight=args.param_loss_weight,
        cont_weight=args.cont_weight,
        categ_weight=args.categ_weight,
        batch_size=args.batch_size,
        mode=mode,
    )
    if not args.eval_only:
        trainer.train(args.epochs, args.resume_from_checkpoint)
    
    best_path = f"{trainer.checkpoint_path}/best_model.pth"
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=DEVICE)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
    trainer.evaluate()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--embedding-dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of transformer attention heads")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--is-audio-encoder", action="store_true", help="Use audio encoder and train in audio_text mode (posterior from audio)")

    # Trainer
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--es-patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--save-interval", type=int, default=10, help="Model save interval (in epochs)")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval (in batches)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # loss
    parser.add_argument("--cont-weight", type=float, default=1.0, help="Weight for continuous parameters loss")
    parser.add_argument("--categ-weight", type=float, default=1.0, help="Weight for categorical parameters loss")
    parser.add_argument("--param-loss-weight", type=float, default=1.0, help="Weight for parameter reconstruction loss (applied to combined cont/categ loss)")
    parser.add_argument("--kl-weight", type=float, default=1.0, help="Weight for KL divergence loss")
    parser.add_argument("--kl-warmup-steps", type=int, default=0, help="Warmup steps for KL weight (0 disables warmup)")

    # others
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    args = parser.parse_args()
    main(args)
