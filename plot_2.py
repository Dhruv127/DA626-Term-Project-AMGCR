import numpy as np
import matplotlib.pyplot as plt

# Valid and test recall and NDCG values for each batch size
valid_recall = {
    1024: [0.0008, 0.0014],
    2048: [0.0008, 0.0014],
    4096: [0.0878, 0.1333],
    10240: [0.0742, 0.1141]
}

valid_ndcg = {
    1024: [0.0004, 0.0006],
    2048: [0.0004, 0.0006],
    4096: [0.0528, 0.0660],
    10240: [0.0438, 0.0555]
}

test_recall = {
    1024: [0.0379, 0.0612],
    2048: [0.0489, 0.0787],
    4096: [0.0894, 0.1361],
    10240: [0.0757, 0.1169]
}

test_ndcg = {
    1024: [0.0275, 0.0353],
    2048: [0.0363, 0.0462],
    4096: [0.0683, 0.0836],
    10240: [0.0579, 0.0714]
}

batch_sizes = list(valid_recall.keys())
x_labels = ['Recall@20', 'Recall@40']

# Set bar width and positions
bar_width = 0.2
indices = np.arange(len(batch_sizes))

# Create separate plots for each metric

# Valid Recall Plot
plt.figure(figsize=(8, 5))
plt.bar(indices, [valid_recall[bs][0] for bs in batch_sizes], width=bar_width, label='Recall@20', align='center')
plt.bar(indices + bar_width, [valid_recall[bs][1] for bs in batch_sizes], width=bar_width, label='Recall@40', align='center')
plt.title('Valid Recall Comparison')
plt.xlabel('Batch Size')
plt.ylabel('Recall Score')
plt.xticks(indices + bar_width / 2, batch_sizes)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Valid NDCG Plot
plt.figure(figsize=(8, 5))
plt.bar(indices, [valid_ndcg[bs][0] for bs in batch_sizes], width=bar_width, label='NDCG@20', align='center')
plt.bar(indices + bar_width, [valid_ndcg[bs][1] for bs in batch_sizes], width=bar_width, label='NDCG@40', align='center')
plt.title('Valid NDCG Comparison')
plt.xlabel('Batch Size')
plt.ylabel('NDCG Score')
plt.xticks(indices + bar_width / 2, batch_sizes)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Test Recall Plot
plt.figure(figsize=(8, 5))
plt.bar(indices, [test_recall[bs][0] for bs in batch_sizes], width=bar_width, label='Recall@20', align='center')
plt.bar(indices + bar_width, [test_recall[bs][1] for bs in batch_sizes], width=bar_width, label='Recall@40', align='center')
plt.title('Test Recall Comparison')
plt.xlabel('Batch Size')
plt.ylabel('Recall Score')
plt.xticks(indices + bar_width / 2, batch_sizes)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Test NDCG Plot
plt.figure(figsize=(8, 5))
plt.bar(indices, [test_ndcg[bs][0] for bs in batch_sizes], width=bar_width, label='NDCG@20', align='center')
plt.bar(indices + bar_width, [test_ndcg[bs][1] for bs in batch_sizes], width=bar_width, label='NDCG@40', align='center')
plt.title('Test NDCG Comparison')
plt.xlabel('Batch Size')
plt.ylabel('NDCG Score')
plt.xticks(indices + bar_width / 2, batch_sizes)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
