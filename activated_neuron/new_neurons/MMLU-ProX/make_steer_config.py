import sys
sys.path.append('/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons')
import pickle
import argparse

import torch

from funcs import (
    unfreeze_pickle
)

def make_steer_config(neuron_list, hidden_size, action="multiply", clamp_value=0.0):
    config = {}
    for layer, neuron in neuron_list:
        hook_name = f"layers.{layer}.mlp.act_fn"
        steering_vector = torch.ones(1, hidden_size)
        steering_vector[0, neuron] = 0.0
        config[hook_name] = {
            "steering_vector": steering_vector,
            "steering_coefficient": clamp_value,
            "action": action,
            "bias": None
        }
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle", type=str, required=True,
                        help="Path to pickle file containing [(layer_idx, neuron_idx), ...]")
    parser.add_argument("--activation_size", type=int, required=True, default=14336,
                        help="MLP activation vector size (default: 14336 for LLaMA-3-8B)")
    parser.add_argument("--output", type=str, required=True, default="steer_config.pt",
                        help="Output .pt filename (default: steer_config.pt)")
    parser.add_argument("--clamp_value", type=float, default=0.0,
                        help="Value to clamp neurons to (default: 0.0)")
    args = parser.parse_args()

    neuron_list = unfreeze_pickle(args.pickle)[:1000]
    config = make_steer_config(
        neuron_list,
        hidden_size=args.activation_size,
        clamp_value=args.clamp_value
    )

    torch.save(config, args.output)
    print(f'Saved steer_config to: {args.output}')

    """ test """
    # config = torch.load("/home/s2410121/proj_LA/activated_neuron/new_neurons/MMLU-ProX/steer_configs/steer_config.pt")

    # for hookpoint, data in config.items():
    #     print(f"Hookpoint: {hookpoint}")
    #     sv = data['steering_vector']
    #     print(f"Steering vector shape: {sv.shape}")
    #     # 特定のニューロンの値を見たいなら
    #     print(f"First 10 values: {sv[0, :10]}")
    #     # ゼロになってるニューロンの位置も確認したい
    #     zero_indices = (sv == 0).nonzero(as_tuple=True)[1]
    #     print(f"Zeroed neuron indices: {zero_indices.tolist()}")
    #     print("------")
    # sys.exit()


# python activated_neuron/new_neurons/MMLU-ProX/make_steer_config.py \
#   --pickle /home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/final_scores/cos_sim/ja_mono_train.pkl \
#   --hidden_size 14336 \
#   --output /home/s2410121/proj_LA/activated_neuron/new_neurons/MMLU-ProX/steer_configs/steer_config.pt \
#   --clamp_value 0.0