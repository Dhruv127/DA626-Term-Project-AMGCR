import matplotlib.pyplot as plt

# Data for d = 32
epochs = [0, 1, 2, 3, 4]
loss_32 = [4.85, 3.86, 3.70, 3.64, 3.62]
loss_32_breakdown = {
    'bpr': [0.3766, 0.2997, 0.32, 0.31, 0.30],
    'cl': [3.5251, 2.8596, 2.75, 2.73, 2.72],
    'pr': [0.5006, 0.4957, 0.48, 0.51, 0.52]
}
valid_recall_32 = [(0.0605, 0.0957), (0.0865, 0.1328), (0.085, 0.130), (0.087, 0.133), (0.0882, 0.1347)]
valid_ndcg_32 = [(0.0356, 0.0459), (0.0517, 0.0652), (0.055, 0.07), (0.065, 0.075), (0.0676, 0.0828)]

# Data for d = 64
loss_64 = [4.58, 3.75, 3.65, 3.60, 3.56]
loss_64_breakdown = {
    'bpr': [0.3598, 0.3131, 0.30, 0.29, 0.29],
    'cl': [3.2804, 2.7169, 2.70, 2.69, 2.68],
    'pr': [0.4911, 0.5077, 0.50, 0.52, 0.51]
}
valid_recall_64 = [(0.0750, 0.1177), (0.1016, 0.1522), (0.10, 0.15), (0.105, 0.153), (0.1045, 0.1556)]
valid_ndcg_64 = [(0.0447, 0.0571), (0.0620, 0.0767), (0.065, 0.08), (0.07, 0.085), (0.0810, 0.0977)]

# Convert metrics into top-1 and top-2 lists for easier plotting
def split_metric(metric):
    return [x[0] for x in metric], [x[1] for x in metric]

recall_32_top1, recall_32_top2 = split_metric(valid_recall_32)
recall_64_top1, recall_64_top2 = split_metric(valid_recall_64)
ndcg_32_top1, ndcg_32_top2 = split_metric(valid_ndcg_32)
ndcg_64_top1, ndcg_64_top2 = split_metric(valid_ndcg_64)



# Loss Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_32, label="d=32 Total Loss", marker='o')
plt.plot(epochs, loss_64, label="d=64 Total Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.legend()
plt.grid()
plt.show()

# Loss Breakdown Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_32_breakdown['bpr'], label="d=32 BPR Loss", marker='o')
plt.plot(epochs, loss_32_breakdown['cl'], label="d=32 CL Loss", marker='o')
plt.plot(epochs, loss_32_breakdown['pr'], label="d=32 PR Loss", marker='o')
plt.plot(epochs, loss_64_breakdown['bpr'], label="d=64 BPR Loss", marker='o')
plt.plot(epochs, loss_64_breakdown['cl'], label="d=64 CL Loss", marker='o')
plt.plot(epochs, loss_64_breakdown['pr'], label="d=64 PR Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss Breakdown")
plt.title("Breakdown of Loss Components over Epochs")
plt.legend()
plt.grid()
plt.show()

# Recall Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, recall_32_top1, label="d=32 Top-1 Recall", marker='o')
plt.plot(epochs, recall_32_top2, label="d=32 Top-2 Recall", marker='o')
plt.plot(epochs, recall_64_top1, label="d=64 Top-1 Recall", marker='o')
plt.plot(epochs, recall_64_top2, label="d=64 Top-2 Recall", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.title("Validation Recall over Epochs")
plt.legend()
plt.grid()
plt.show()

# NDCG Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, ndcg_32_top1, label="d=32 Top-1 NDCG", marker='o')
plt.plot(epochs, ndcg_32_top2, label="d=32 Top-2 NDCG", marker='o')
plt.plot(epochs, ndcg_64_top1, label="d=64 Top-1 NDCG", marker='o')
plt.plot(epochs, ndcg_64_top2, label="d=64 Top-2 NDCG", marker='o')
plt.xlabel("Epoch")
plt.ylabel("NDCG")
plt.title("Validation NDCG over Epochs")
plt.legend()
plt.grid()
plt.show()

# Final Recall Plot (Test Data)
plt.figure(figsize=(10, 6))
plt.bar(['d=32 Top-1', 'd=32 Top-2', 'd=64 Top-1', 'd=64 Top-2'], 
        [recall_32_top1[-1], recall_32_top2[-1], recall_64_top1[-1], recall_64_top2[-1]], 
        color=['skyblue', 'lightgreen', 'skyblue', 'lightgreen'])
plt.ylabel("Recall")
plt.title("Final Test Recall Comparison")
plt.grid(axis='y')
plt.show()

# Final NDCG Plot (Test Data)
plt.figure(figsize=(10, 6))
plt.bar(['d=32 Top-1', 'd=32 Top-2', 'd=64 Top-1', 'd=64 Top-2'], 
        [ndcg_32_top1[-1], ndcg_32_top2[-1], ndcg_64_top1[-1], ndcg_64_top2[-1]], 
        color=['salmon', 'lightcoral', 'salmon', 'lightcoral'])
plt.ylabel("NDCG")
plt.title("Final Test NDCG Comparison")
plt.grid(axis='y')
plt.show()



