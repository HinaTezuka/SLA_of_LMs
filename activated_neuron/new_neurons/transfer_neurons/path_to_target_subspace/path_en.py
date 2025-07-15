import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
from collections import defaultdict

import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.backends.backend_pdf import PdfPages

from funcs import (
    monolingual_dataset,
    compute_scores,
    compute_scores_optimized,
    sort_neurons_by_score,
    save_as_pickle,
    unfreeze_pickle,
)

model_names = ["meta-llama/Meta-Llama-3-8B", "mistralai/Mistral-7B-v0.3", "CohereForAI/aya-expanse-8b"]
device = "cuda" if torch.cuda.is_available() else "cpu"
langs = ["ja", "nl", "ko", "it"]
middle_layer = 10 # 10 layer.
results = defaultdict(list)
model_name_MAP = {
    'llama3': 'LLaMA3-8B',
    'mistral': 'Mistral-7B',
    'aya': 'Aya expanse-8B'
}

# for centroids of en-subspaces.
plt.rcParams["font.family"] = "DejaVu Serif"
figure, ax = plt.subplots(figsize=(10, 10))
for model_type in ['llama3', 'mistral', 'aya']:
    # centroids(en-only).
    c_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/centroids/c_train_en.pkl"
    centroids = unfreeze_pickle(c_path)
    path_to_middle = centroids[middle_layer] - centroids[0]

    for layer_i in range(1, middle_layer):
        path_corrent = centroids[layer_i] - centroids[layer_i - 1]
        sim_score = cosine_similarity(path_to_middle.reshape(1, -1), path_corrent.reshape(1, -1))
        results[model_type].append(sim_score.item())
    
    # visualize with lineplot
    x_vals = range(2, middle_layer+1)
    ax.plot(x_vals, results[model_type], '-p', linewidth=2, markersize=8, label=model_name_MAP[model_type])
ax.set_title('Similarity of en-Centroids Trajectory', fontsize=35)
xticks = [1] + list(range(5, middle_layer + 1, 5))
xticks = sorted(set(xticks))  # 重複を避けつつ昇順に
ax.set_xticks(xticks)
ax.tick_params(axis='both', labelsize=20)
ax.set_xlabel('Layers', fontsize=35)
ax.set_ylabel('Cosine Sim', fontsize=35)
ax.set_ylim(0, 1)
ax.legend(fontsize=35)

save_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/images/transfers/subspace/vec_trajectory/mid_{middle_layer}_en_centroids'
with PdfPages(save_path + '.pdf') as pdf:
    pdf.savefig(bbox_inches='tight', pad_inches=0.01)
    plt.close()



# for every L2.
# for model_type in ['llama3', 'mistral', 'aya']:
#     # calc scores.
#     for L2 in langs:
#         # monolingual_sentences = monolingual_dataset(L2, num_sentences)
#         monolingual_sentences = unfreeze_pickle(f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/sentence_data/{L2}_mono_train.pkl")
#         # centroids(en-only).
#         c_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/centroids/c_train_en.pkl"
#         centroids = unfreeze_pickle(c_path)