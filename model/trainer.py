import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging
from config import DEVICE
from model.loss.param_loss import ParamsLoss
from model.loss.kl_loss import KLDivergenceLoss
from torch.utils.data import DataLoader
import tqdm
from utils.plot import plot_losses
import os

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataset,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        batch_size: int = 32,
        checkpoint_path: str = "checkpoints",
        log_interval: int = 10,
        early_stopping_patience: int = 10,
        logger: Optional[logging.Logger] = None,
        kl_weight: float = 1.0,
        kl_warmup_steps: int = 0,
        param_loss_weight: float = 1.0,
        cont_weight: float = 1.0,
        categ_weight: float = 1.0,
        mode: str = "text_only",
    ):
        self.batch_size = batch_size
        self.model = model
        self.train_dataloader = DataLoader(dataset.dataset['train'], batch_size=self.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
        self.val_dataloader = DataLoader(dataset.dataset['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
        self.test_dataloader = DataLoader(dataset.dataset['test'], batch_size=self.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_path = checkpoint_path
        self.log_interval = log_interval
        self.early_stopping_patience = early_stopping_patience
        self.logger = logger or logging.getLogger(__name__)
        self.kl_weight = kl_weight
        self.kl_warmup_steps = kl_warmup_steps
        self.param_loss_weight = param_loss_weight
        self.cont_weight = cont_weight
        self.categ_weight = categ_weight
        self.mode = mode
        self.device = DEVICE
        self.best_loss = float('inf')
        self.param_train_losses = []
        self.param_val_losses = []
        self.kl_train_losses = []
        self.kl_val_losses = []
        self.global_step = 0

        self.model.to(self.device)
        self.param_loss_fn = ParamsLoss(cont_weight=self.cont_weight, categ_weight=self.categ_weight)
        self.kl_loss_fn = KLDivergenceLoss()

    def calculate_loss(self, 
      decoder_outputs,
      params_batch,
      prior : Optional[dict] = None,
      posterior : Optional[dict] = None,
      latent : Optional[torch.Tensor] = None,
      mode : Optional[str] = None,
      is_validation: bool = False,
    ):
      if mode == None:
        mode = self.mode
      param_loss, categ_loss, cont_loss = self.param_loss_fn(
        categ_pred=decoder_outputs["categorical"],
        categ_target=params_batch['categ'],
        cont_pred=decoder_outputs["continuous"],
        cont_target=params_batch['cont'],
      )
      if mode == "audio_text" and posterior is not None and prior is not None:
          kl_raw = self.kl_loss_fn(
            posterior["mu"],
            posterior["logvar"],
            prior["mu"],
            prior["logvar"],
          )
          # バリデーション時はwarmupを考慮せず、kl_weightをそのまま使用
          if is_validation:
            warmup_scale = 1.0
          else:
            warmup_scale = 1.0
            if self.kl_warmup_steps and self.kl_warmup_steps > 0:
              warmup_scale = min(1.0, self.global_step / float(self.kl_warmup_steps))
          kl_loss = self.kl_weight * warmup_scale * kl_raw
      else:
          kl_raw = torch.zeros((), device=self.device)
          kl_loss = kl_raw
      total_loss = self.param_loss_weight * param_loss + kl_loss
      return total_loss, param_loss, categ_loss, cont_loss, kl_raw, kl_loss\

    def train(self, num_epochs: int, resume_from_checkpoint: Optional[str] = None):
        start_epoch = 0
        if resume_from_checkpoint:
            start_epoch = self._load_checkpoint(resume_from_checkpoint)
            self.logger.info(f"Resumed training from checkpoint: {resume_from_checkpoint} at epoch {start_epoch}")
        
        for epoch in range(start_epoch, num_epochs):
            self.model.train()
            epoch_param_loss = 0.0
            epoch_kl_loss = 0.0
            early_stopping_counter = 0

            train_bar = tqdm.tqdm(total=len(self.train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", postfix="total_loss=0.0, categ_loss=0.0, cont_loss=0.0")
            for batch in self.train_dataloader:
                self.optimizer.zero_grad()
                texts_batch, audio_inputs, _, params_batch = batch
                for key in params_batch['categ']:
                    params_batch['categ'][key] = params_batch['categ'][key].to(DEVICE)
                for key in params_batch['cont']:
                    params_batch['cont'][key] = params_batch['cont'][key].to(DEVICE)
                
                outputs = self.model(
                    text_inputs=texts_batch,
                    audio_inputs=audio_inputs,
                    mode=self.mode,
                )
                decoder_out = outputs["decoder_outputs"]
                posterior = outputs["posterior"]
                prior = outputs["prior"]
                letent = outputs["latent"]

                total_loss, param_loss, categ_loss, cont_loss, kl_raw, kl_loss = self.calculate_loss(
                    decoder_outputs=decoder_out,
                    params_batch=params_batch,
                    prior=prior,
                    posterior=posterior,
                    latent=letent
                )
                epoch_kl_loss += kl_raw.item()
                epoch_param_loss += param_loss.item()
                total_loss.backward()
                self.optimizer.step()
                self.global_step += 1
                train_bar.set_postfix(total_loss=f"{total_loss.item():.4f}", categ_loss=f"{categ_loss.item():.4f}", cont_loss=f"{cont_loss.item():.4f}", kl_loss=f"{kl_loss.item():.4f} ({kl_raw.item():.4f})")
                train_bar.update(1)
            train_bar.close()
            val_loss = self.validate()
            self.param_train_losses.append(epoch_param_loss/len(self.train_dataloader))
            self.kl_train_losses.append(epoch_kl_loss/len(self.train_dataloader))
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self._save_checkpoint(epoch, is_best=True)
            else:
                early_stopping_counter += 1
            
            plot_losses(self.param_train_losses, self.param_val_losses, f"{self.checkpoint_path}/param_loss_curve.png")
            plot_losses(self.kl_train_losses, self.kl_val_losses, f"{self.checkpoint_path}/kl_loss_curve.png")
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_param_loss/len(self.train_dataloader)}, Val Loss: {val_loss}")
            if epoch % self.save_interval == 0:
                self._save_checkpoint(epoch, is_best=False)
            if early_stopping_counter >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            

    def validate(self):
      self.model.eval()
      epoch_param_loss = 0.0
      epoch_kl_loss = 0.0
      epoch_total_loss = 0.0
      with torch.no_grad():
        for batch in self.val_dataloader:
          texts_batch, audio_inputs, _, params_batch = batch
          for key in params_batch['categ']:
            params_batch['categ'][key] = params_batch['categ'][key].to(DEVICE)
          for key in params_batch['cont']:
            params_batch['cont'][key] = params_batch['cont'][key].to(DEVICE)
          outputs = self.model(
            text_inputs=texts_batch,
            audio_inputs=audio_inputs,
            mode=self.mode,
          )
          decoder_out = outputs["decoder_outputs"]
          posterior = outputs["posterior"]
          prior = outputs["prior"]
          letent = outputs["latent"]
          
          total_loss, param_loss, categ_loss, cont_loss, kl_raw, kl_loss = self.calculate_loss(
              decoder_outputs=decoder_out,
              params_batch=params_batch,
              prior=prior,
              posterior=posterior,
              latent=letent,
              is_validation=True,
          )
          epoch_param_loss += param_loss.item()
          epoch_kl_loss += kl_raw.item()
          epoch_total_loss += total_loss.item()

      self.param_val_losses.append(epoch_param_loss/len(self.val_dataloader))
      self.kl_val_losses.append(epoch_kl_loss/len(self.val_dataloader))
      return epoch_total_loss/len(self.val_dataloader)

    def evaluate(self) -> None:
        self.model.eval()
        data_loader = self.test_dataloader

        # text_only mode用の累積変数
        text_only_total_loss = 0.0
        text_only_param_loss = 0.0
        text_only_kl_raw = 0.0
        text_only_cont_mae = 0.0
        text_only_cont_count = 0
        text_only_categ_accuracies = {}
        text_only_categ_totals = {}

        # audio_text mode用の累積変数（modeがaudio_textの場合のみ使用）
        audio_text_total_loss = 0.0
        audio_text_param_loss = 0.0
        audio_text_kl_raw = 0.0
        audio_text_cont_mae = 0.0
        audio_text_cont_count = 0
        audio_text_categ_accuracies = {}
        audio_text_categ_totals = {}

        with torch.no_grad():
            for batch in tqdm.tqdm(data_loader, desc="Detailed Evaluation"):
                texts_batch, audio_inputs, _, params_batch = batch
                for key in params_batch['categ']:
                    params_batch['categ'][key] = params_batch['categ'][key].to(DEVICE)
                for key in params_batch['cont']:
                    params_batch['cont'][key] = params_batch['cont'][key].to(DEVICE)

                # text_only modeでの評価
                text_only_outputs = self.model(
                    text_inputs=texts_batch,
                    mode="text_only",
                )
                text_only_total, text_only_param, _, _, text_only_kl_r, _ = self.calculate_loss(
                    decoder_outputs=text_only_outputs["decoder_outputs"],
                    params_batch=params_batch,
                    prior=text_only_outputs["prior"],
                    posterior=text_only_outputs["posterior"],
                    latent=text_only_outputs["latent"],
                    mode="text_only",
                    is_validation=True,
                )
                text_only_total_loss += text_only_total.item()
                text_only_param_loss += text_only_param.item()
                text_only_kl_raw += text_only_kl_r.item()

                # ContinuousパラメータのMAE計算（text_only）
                cont_pred = text_only_outputs["decoder_outputs"]["continuous"]
                cont_target = params_batch['cont']
                for param_name in cont_pred.keys():
                    if param_name in cont_target:
                        pred_values = cont_pred[param_name]
                        true_values = cont_target[param_name]
                        text_only_cont_mae += torch.abs(pred_values - true_values).sum().item()
                        text_only_cont_count += true_values.numel()

                # CategoricalパラメータのAccuracy計算（text_only）
                categ_pred = text_only_outputs["decoder_outputs"]["categorical"]
                categ_target = params_batch['categ']
                for param_name in categ_target.keys():
                    if param_name in categ_pred:
                        pred_labels = torch.argmax(categ_pred[param_name], dim=1)
                        true_labels = categ_target[param_name]
                        correct = (pred_labels == true_labels).sum().item()
                        total = true_labels.size(0)
                        if param_name not in text_only_categ_accuracies:
                            text_only_categ_accuracies[param_name] = 0
                        if param_name not in text_only_categ_totals:
                            text_only_categ_totals[param_name] = 0
                        text_only_categ_accuracies[param_name] += correct
                        text_only_categ_totals[param_name] += total

                # audio_text modeでの評価（modeがaudio_textの場合のみ）
                if self.mode == "audio_text":
                    audio_text_outputs = self.model(
                        text_inputs=texts_batch,
                        audio_inputs=audio_inputs,
                        mode="audio_text",
                    )
                    audio_text_total, audio_text_param, _, _, audio_text_kl_r, _ = self.calculate_loss(
                        decoder_outputs=audio_text_outputs["decoder_outputs"],
                        params_batch=params_batch,
                        prior=audio_text_outputs["prior"],
                        posterior=audio_text_outputs["posterior"],
                        latent=audio_text_outputs["latent"],
                        mode="audio_text",
                        is_validation=True,
                    )
                    audio_text_total_loss += audio_text_total.item()
                    audio_text_param_loss += audio_text_param.item()
                    audio_text_kl_raw += audio_text_kl_r.item()

                    # ContinuousパラメータのMAE計算（audio_text）
                    cont_pred = audio_text_outputs["decoder_outputs"]["continuous"]
                    cont_target = params_batch['cont']
                    for param_name in cont_pred.keys():
                        if param_name in cont_target:
                            pred_values = cont_pred[param_name]
                            true_values = cont_target[param_name]
                            audio_text_cont_mae += torch.abs(pred_values - true_values).sum().item()
                            audio_text_cont_count += true_values.numel()

                    # CategoricalパラメータのAccuracy計算（audio_text）
                    categ_pred = audio_text_outputs["decoder_outputs"]["categorical"]
                    categ_target = params_batch['categ']
                    for param_name in categ_target.keys():
                        if param_name in categ_pred:
                            pred_labels = torch.argmax(categ_pred[param_name], dim=1)
                            true_labels = categ_target[param_name]
                            correct = (pred_labels == true_labels).sum().item()
                            total = true_labels.size(0)
                            if param_name not in audio_text_categ_accuracies:
                                audio_text_categ_accuracies[param_name] = 0
                            if param_name not in audio_text_categ_totals:
                                audio_text_categ_totals[param_name] = 0
                            audio_text_categ_accuracies[param_name] += correct
                            audio_text_categ_totals[param_name] += total

        # text_only modeの結果集計と出力
        num_batches = len(data_loader)
        text_only_avg_total_loss = text_only_total_loss / num_batches
        text_only_avg_param_loss = text_only_param_loss / num_batches
        text_only_avg_kl_raw = text_only_kl_raw / num_batches
        text_only_cont_mae_avg = text_only_cont_mae / text_only_cont_count if text_only_cont_count > 0 else 0.0
        text_only_categ_correct = sum(text_only_categ_accuracies.values())
        text_only_categ_total = sum(text_only_categ_totals.values())
        text_only_categ_overall_acc = text_only_categ_correct / text_only_categ_total if text_only_categ_total > 0 else 0.0

        self.logger.info("=" * 80)
        self.logger.info("Detailed Evaluation Results (text_only mode)")
        self.logger.info("=" * 80)
        self.logger.info(f"Total Avg Loss: {text_only_avg_total_loss:.4f}")
        self.logger.info(f"Param Loss: {text_only_avg_param_loss:.4f}")
        self.logger.info(f"KL Raw: {text_only_avg_kl_raw:.4f}")
        self.logger.info(f"Continuous Params MAE: {text_only_cont_mae_avg:.4f}")
        self.logger.info(f"Categorical Params Overall Accuracy: {text_only_categ_overall_acc:.4f}")
        for param_name in sorted(text_only_categ_accuracies.keys()):
            accuracy = text_only_categ_accuracies[param_name] / text_only_categ_totals[param_name] if text_only_categ_totals[param_name] > 0 else 0.0
            self.logger.info(f"  {param_name}: {accuracy:.4f}")

        # audio_text modeの結果集計と出力（modeがaudio_textの場合のみ）
        if self.mode == "audio_text":
            audio_text_avg_total_loss = audio_text_total_loss / num_batches
            audio_text_avg_param_loss = audio_text_param_loss / num_batches
            audio_text_avg_kl_raw = audio_text_kl_raw / num_batches
            audio_text_cont_mae_avg = audio_text_cont_mae / audio_text_cont_count if audio_text_cont_count > 0 else 0.0
            audio_text_categ_correct = sum(audio_text_categ_accuracies.values())
            audio_text_categ_total = sum(audio_text_categ_totals.values())
            audio_text_categ_overall_acc = audio_text_categ_correct / audio_text_categ_total if audio_text_categ_total > 0 else 0.0

            self.logger.info("=" * 80)
            self.logger.info("Detailed Evaluation Results (audio_text mode)")
            self.logger.info("=" * 80)
            self.logger.info(f"Total Avg Loss: {audio_text_avg_total_loss:.4f}")
            self.logger.info(f"Param Loss: {audio_text_avg_param_loss:.4f}")
            self.logger.info(f"KL Raw: {audio_text_avg_kl_raw:.4f}")
            self.logger.info(f"Continuous Params MAE: {audio_text_cont_mae_avg:.4f}")
            self.logger.info(f"Categorical Params Overall Accuracy: {audio_text_categ_overall_acc:.4f}")
            for param_name in sorted(audio_text_categ_accuracies.keys()):
                accuracy = audio_text_categ_accuracies[param_name] / audio_text_categ_totals[param_name] if audio_text_categ_totals[param_name] > 0 else 0.0
                self.logger.info(f"  {param_name}: {accuracy:.4f}")

        # 結果をファイルに保存
        os.makedirs(self.checkpoint_path, exist_ok=True)
        with open(f"{self.checkpoint_path}/detailed_evaluation.txt", "w") as f:
            f.write("Detailed Evaluation Results\n")
            f.write(f"Model: {self.model.__class__.__name__} (heads={self.model.num_heads}, layers={self.model.num_layers}, dropout={self.model.dropout}, embed_dim={self.model.embed_dim})\n")
            f.write(f"Mode: {self.mode}\n")
            f.write("=" * 80 + "\n\n")
            f.write("text_only mode:\n")
            f.write(f"Total Avg Loss: {text_only_avg_total_loss:.4f}\n")
            f.write(f"Param Loss: {text_only_avg_param_loss:.4f}\n")
            f.write(f"KL Raw: {text_only_avg_kl_raw:.4f}\n")
            f.write(f"Continuous Params MAE: {text_only_cont_mae_avg:.4f}\n")
            f.write(f"Categorical Params Overall Accuracy: {text_only_categ_overall_acc:.4f}\n")
            for param_name in sorted(text_only_categ_accuracies.keys()):
                accuracy = text_only_categ_accuracies[param_name] / text_only_categ_totals[param_name] if text_only_categ_totals[param_name] > 0 else 0.0
                f.write(f"Categorical Param: {param_name}, Accuracy: {accuracy:.4f}\n")
            
            if self.mode == "audio_text":
                f.write("\n" + "=" * 80 + "\n\n")
                f.write("audio_text mode:\n")
                f.write(f"Total Avg Loss: {audio_text_avg_total_loss:.4f}\n")
                f.write(f"Param Loss: {audio_text_avg_param_loss:.4f}\n")
                f.write(f"KL Raw: {audio_text_avg_kl_raw:.4f}\n")
                f.write(f"Continuous Params MAE: {audio_text_cont_mae_avg:.4f}\n")
                f.write(f"Categorical Params Overall Accuracy: {audio_text_categ_overall_acc:.4f}\n")
                for param_name in sorted(audio_text_categ_accuracies.keys()):
                    accuracy = audio_text_categ_accuracies[param_name] / audio_text_categ_totals[param_name] if audio_text_categ_totals[param_name] > 0 else 0.0
                    f.write(f"Categorical Param: {param_name}, Accuracy: {accuracy:.4f}\n")
        
        self.logger.info(f"Detailed evaluation results saved to {self.checkpoint_path}/detailed_evaluation.txt")

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        path: str = f"{self.checkpoint_path}/checkpoint_epoch_{epoch+1}.pth" if not is_best else f"{self.checkpoint_path}/best_model.pth"

        os.makedirs(self.checkpoint_path, exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'param_train_losses': self.param_train_losses,
            'param_val_losses': self.param_val_losses,
            'kl_train_losses': self.kl_train_losses,
            'kl_val_losses': self.kl_val_losses,
        }, path)
        if is_best:
            self.logger.info(f"Saved best model checkpoint to {path}")

    def _load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.param_train_losses = checkpoint.get('param_train_losses', [])
        self.param_val_losses = checkpoint.get('param_val_losses', [])
        self.kl_train_losses = checkpoint.get('kl_train_losses', [])
        self.kl_val_losses = checkpoint.get('kl_val_losses', [])
        self.global_step = checkpoint.get('global_step', 0)
        return checkpoint['epoch']
      