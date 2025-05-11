import time
import pandas as pd

class TrainingLogger:
    def __init__(self):
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_recall": [],
            "val_ndcg": [],
            "epoch_time_s": []
        }
        self._start_time = None

    def start_epoch(self):
        self._start_time = time.time()

    def log_epoch(self, epoch: int, train_loss: float, val_recall: float, val_ndcg: float):
        if self._start_time is None:
            raise RuntimeError("start time is required for each epoch")
        duration = time.time() - self._start_time
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_recall"].append(val_recall)
        self.history["val_ndcg"].append(val_ndcg)
        self.history["epoch_time_s"].append(duration)
        self._start_time = None

    def save(self, filepath: str = "training_history.csv"):
        # collected data into csv file
        df = pd.DataFrame(self.history)
        df.to_csv(filepath, index=False)
