import matplotlib.pyplot as plt

def plot_losses(train_losses: list, val_losses: list, save_path: str = "loss_curve.png"):
  plt.figure(figsize=(10, 5))
  plt.plot(train_losses, label='Train Loss')
  plt.plot(val_losses, label='Val Loss')
  plt.legend(loc='upper right')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Loss Curve')
  plt.savefig(save_path)
  plt.close()