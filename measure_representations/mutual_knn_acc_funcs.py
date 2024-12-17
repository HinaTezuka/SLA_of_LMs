"""
some funcs were copied from https://github.com/minyoungg/platonic-rep/blob/main/metrics.py
The Platonic Representation Hypothesis:https://arxiv.org/abs/2405.07987
"""

from datasets import load_dataset
import torch
import torch.nn.functional as F

# Extract features from the models (example for embedding layer or first hidden layer)
def extract_representations(model, tokenizer, texts, layer_idx=-1) -> torch.Tensor: # layer_indexで「どの層」のrepresentationsをとるかを指定
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
        hidden_states = outputs.hidden_states  # List of layer-wise hidden states
        print(hidden_states)
        representations = hidden_states[layer_idx]  # Extract the chosen layer
        print(representations)
    # return representations.mean(dim=1)  # Mean-pooling across tokens
    return F.normalize(representations.mean(dim=1))

def mutual_knn(feats_A, feats_B, topk):
    """
    Computes the mutual KNN accuracy.

    Args:
        feats_A: A torch tensor of shape N x feat_dim
        feats_B: A torch tensor of shape N x feat_dim

    Returns:
        A float representing the mutual KNN accuracy
    """
    knn_A = compute_nearest_neighbors(feats_A, topk)
    knn_B = compute_nearest_neighbors(feats_B, topk)

    n = knn_A.shape[0]
    topk = knn_A.shape[1]

    # Create a range tensor for indexing
    range_tensor = torch.arange(n, device=knn_A.device).unsqueeze(1)

    # Create binary masks for knn_A and knn_B
    lvm_mask = torch.zeros(n, n, device=knn_A.device)
    llm_mask = torch.zeros(n, n, device=knn_A.device)

    lvm_mask[range_tensor, knn_A] = 1.0
    llm_mask[range_tensor, knn_B] = 1.0

    acc = (lvm_mask * llm_mask).sum(dim=1) / topk

    return acc.mean().item()

def compute_nearest_neighbors(feats, topk=5):
    """
    Compute the nearest neighbors of feats
    Args:
        feats: a torch tensor of shape N x D
        topk: the number of nearest neighbors to return
    Returns:
        knn: a torch tensor of shape N x topk
    """
    assert feats.ndim == 2, f"Expected feats to be 2D, got {feats.ndim}"
    knn = (
        (feats @ feats.T).fill_diagonal_(-1e8).argsort(dim=1, descending=True)[:, :topk]
    )
    return knn

# datasetからen, L2の対訳textを取得
def get_texts_from_translation_corpus(n_samples, L2_iso_code, dataset="tatoeba"):
    if dataset == "tatoeba":
        dataset = load_dataset(dataset, lang1="en", lang2=L2_iso_code, split="train")
    elif dataset == "en_ger":
        dataset = load_dataset("KarthikSaran/trans_en_ger", split="train")
    else:
        print("dataset is not defined!")
    # print(dataset[0])
    texts_en = [sample['translation']['en'] for sample in dataset.select(range(n_samples))]
    texts_L2 = [sample['translation'][L2_iso_code] for sample in dataset.select(range(n_samples))]

    return texts_en, texts_L2

# mutual_knn_accuracyを算出
def compute_mutual_knn_acc(
    model_base,
    model_L2,
    model_base_tokenizer,
    model_L2_tokenizer,
    texts_en,
    texts_L2,
    topk
    ):
    # representationsを抽出
    feats_base = extract_representations(model_base, model_base_tokenizer, texts_en)
    feats_L2 = extract_representations(model_L2, model_L2_tokenizer, texts_L2)
    print(feats_base, '\n', feats_L2)
    mutual_knn_accuracy = mutual_knn(feats_base, feats_L2, topk)

    return mutual_knn_accuracy
