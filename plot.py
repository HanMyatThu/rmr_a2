import pandas as pd
import matplotlib.pyplot as plt

# Load training history for different sequence lengths
hist_seq10 = pd.read_csv("./ablation/layers/2/training_history.csv")
hist_seq20 = pd.read_csv("./ablation/layers/4/training_history.csv")
hist_seq50 = pd.read_csv("./ablation/layers/6/training_history.csv")

# Plot Validation NDCG@10 vs Epoch
plt.figure()
plt.plot(hist_seq10["epoch"], hist_seq10["val_ndcg"], label="Layers=2")
plt.plot(hist_seq20["epoch"], hist_seq20["val_ndcg"], label="Layers=4")
plt.plot(hist_seq50["epoch"], hist_seq50["val_ndcg"], label="Layers=6")
plt.xlabel("Epoch")
plt.ylabel("Validation NDCG@10")
plt.legend()
plt.title("Effect of Numbers of Layer on Validation NDCG@10")
plt.savefig("layers_ndcg.png")
plt.show()

# Plot Validation Recall@10 vs Epoch
plt.figure()
plt.plot(hist_seq10["epoch"], hist_seq10["val_recall"], label="Layers=2")
plt.plot(hist_seq20["epoch"], hist_seq20["val_recall"], label="Layers=4")
plt.plot(hist_seq50["epoch"], hist_seq50["val_recall"], label="Layers=6")
plt.xlabel("Epoch")
plt.ylabel("Validation Recall@10")
plt.legend()
plt.title("Effect of Numbers of Layer on Validation Recall@10")
plt.savefig("layers_recall.png")
plt.show()

# Plot Epoch Time vs Epoch
plt.figure()
plt.plot(hist_seq10["epoch"], hist_seq10["epoch_time_s"], label="Layers=2")
plt.plot(hist_seq20["epoch"], hist_seq20["epoch_time_s"], label="Layers=4")
plt.plot(hist_seq50["epoch"], hist_seq50["epoch_time_s"], label="Layers=6")
plt.xlabel("Epoch")
plt.ylabel("Epoch Time (s)")
plt.legend()
plt.title("Training Time per Epoch for Different Layer Numbers")
plt.savefig("layers_time.png")
plt.show()
