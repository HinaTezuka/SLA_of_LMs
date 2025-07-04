import numpy as np
import json


model_types = ['aya', 'mistral', 'llama3']

"""  """
for model_type in model_types:
    l1_p = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons/qa/output_samples/{model_type}/confidence_scores/entropy/ja_normal.json'
    with open(l1_p, "r", encoding="utf-8") as f:
        l1 = json.load(f)

    l1 = np.array(l1)

    for L2 in ['ja', 'nl', 'ko']:
        l2_p = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons/qa/output_samples/{model_type}/confidence_scores/entropy/input_ja_deact_{L2}_distribution.json'
        with open(l2_p, "r", encoding="utf-8") as f:
            l2 = json.load(f)
        l2 = np.array(l2)

        # 要素ごとに比較し、l2 の方が大きい割合を計算
        proportion_l2_higher = np.mean(l2 > l1)

        print(f"{model_type}, {L2} — ave.: normal: {l1.mean():.4f}, input_ja, deactivate_{L2}: {l2.mean():.4f}, deactivation > normal: {proportion_l2_higher:.2%}")

""" perplexity """