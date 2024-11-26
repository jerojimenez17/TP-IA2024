import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo CSV
data = pd.read_csv("results.csv")

# Graficar pérdidas de entrenamiento y validación
plt.figure(figsize=(10, 6))
plt.plot(data["epoch"], data["train/box_loss"], label="Train Box Loss")
plt.plot(data["epoch"], data["val/box_loss"], label="Validation Box Loss")
plt.plot(data["epoch"], data["train/cls_loss"], label="Train Class Loss")
plt.plot(data["epoch"], data["val/cls_loss"], label="Validation Class Loss")
plt.plot(data["epoch"], data["train/dfl_loss"], label="Train DFL Loss")
plt.plot(data["epoch"], data["val/dfl_loss"], label="Validation DFL Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Losses")
plt.legend()
plt.grid()
plt.show()

# Graficar métricas de precisión, recall y mAP
plt.figure(figsize=(10, 6))
plt.plot(data["epoch"], data["metrics/precision(B)"], label="Precision")
plt.plot(data["epoch"], data["metrics/recall(B)"], label="Recall")
plt.plot(data["epoch"], data["metrics/mAP50(B)"], label="mAP@50")
plt.plot(data["epoch"], data["metrics/mAP50-95(B)"], label="mAP@50-95")
plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.title("Metrics (Precision, Recall, mAP)")
plt.legend()
plt.grid()
plt.show()

# Graficar tasas de aprendizaje
plt.figure(figsize=(10, 6))
plt.plot(data["epoch"], data["lr/pg0"], label="Learning Rate pg0")
plt.plot(data["epoch"], data["lr/pg1"], label="Learning Rate pg1")
plt.plot(data["epoch"], data["lr/pg2"], label="Learning Rate pg2")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rates")
plt.legend()
plt.grid()
plt.show()
