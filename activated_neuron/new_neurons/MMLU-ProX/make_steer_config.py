import sys
sys.path.append('/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons')
import pickle
import argparse
import random

import torch

from funcs import (
    unfreeze_pickle
)

# def make_steer_config(neuron_list, hidden_size, action="multiply", clamp_value=0.0):
#     config = {}
#     for layer, neuron in neuron_list:
#         hook_name = f"layers.{layer}.mlp.act_fn"
#         steering_vector = torch.ones(1, hidden_size)
#         steering_vector[0, neuron] = 0.0
#         config[hook_name] = {
#             "steering_vector": steering_vector,
#             "steering_coefficient": clamp_value,
#             "action": action,
#             "bias": None
#         }
#     return config

def make_steer_config(neuron_list, hidden_size, action="multiply", clamp_value=0.0):
    config = {}
    for layer, neuron in neuron_list:
        hook_name = f"layers.{layer}.mlp.act_fn"
        
        # If this layer is not yet in config, initialize a new entry
        if hook_name not in config:
            steering_vector = torch.ones(1, hidden_size)
            config[hook_name] = {
                "steering_vector": steering_vector,
                "steering_coefficient": clamp_value,
                "action": action,
                "bias": None
            }
        
        # Set the corresponding neuron position to 0.0
        config[hook_name]["steering_vector"][0, neuron] = 0.0

    return config

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--pickle", type=str, required=True,
    #                     help="Path to pickle file containing [(layer_idx, neuron_idx), ...]")
    # parser.add_argument("--activation_size", type=int, required=True, default=14336,
    #                     help="MLP activation vector size (default: 14336 for LLaMA-3-8B)")
    # parser.add_argument("--output", type=str, required=True, default="steer_config.pt",
    #                     help="Output .pt filename (default: steer_config.pt)")
    # parser.add_argument("--clamp_value", type=float, default=0.0,
    #                     help="Value to clamp neurons to (default: 0.0)")
    # args = parser.parse_args()

    # neuron_list = unfreeze_pickle(args.pickle)[:1000]
    # config = make_steer_config(
    #     neuron_list,
    #     hidden_size=args.activation_size,
    #     clamp_value=args.clamp_value
    # )

    # torch.save(config, args.output)
    # print(f'Saved steer_config to: {args.output}')

# python activated_neuron/new_neurons/MMLU-ProX/make_steer_config.py \
#   --pickle /home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/llama3/final_scores/cos_sim/ja_mono_train.pkl \
#   --hidden_size 14336 \
#   --output /home/s2410121/proj_LA/activated_neuron/new_neurons/MMLU-ProX/steer_configs/steer_config.pt \
#   --clamp_value 0.0


    """ all """
    # langs = ['ja', 'nl', 'ko', 'it', 'vi', 'ru', 'fr']
    model_types = ['llama3', 'mistral', 'aya']
    langs = ['ja', 'ko', 'fr'] # for MMLU-ProX
    score_type = 'cos_sim'
    intervention_num = 1000
    action_type = 'multiply'
    
    for model_type in model_types:
        neuron_num = 14336 if model_type in ['llama3', 'mistral', 'aya'] else 10240 # 10240: BLOOM
        for L2 in langs:
            # # get type-1 neurons.
            # type1_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/{score_type}/{L2}_mono_train.pkl"
            # type1_neurons = unfreeze_pickle(type1_path)

            # # type-1 neurons.
            # type1_neurons_main = type1_neurons[:intervention_num]

            # get type-2 neurons.
            type1_path = f"/home/s2410121/proj_LA/activated_neuron/new_neurons/pickles/transfer_neurons/{model_type}/final_scores/reverse/{score_type}/{L2}_sorted_neurons.pkl"
            type1_neurons = unfreeze_pickle(type1_path) # type-2
            type1_neurons = [neuron for neuron in type1_neurons if neuron[0] in [ _ for _ in range(20, 32)]]

            # type-1 neurons.
            type1_neurons_main = type1_neurons[:intervention_num]

            # baseline.
            random.seed(42)
            type1_neurons_baseline = random.sample(type1_neurons[intervention_num:], intervention_num)

            config = make_steer_config(
                # type1_neurons_main, # type-1/2
                type1_neurons_baseline, # baseline
                hidden_size=neuron_num,
                action=action_type,
                clamp_value=0.0
            )

            # type-1
            # output_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/MMLU-ProX/steer_configs/{model_type}/{score_type}/{L2}.pt'
            # output_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/MMLU-ProX/steer_configs/{model_type}/{score_type}/{L2}_baseline.pt'
            #
            # type-2
            # output_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/MMLU-ProX/steer_configs/{model_type}/{score_type}/{L2}_type2.pt'
            output_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/MMLU-ProX/steer_configs/{model_type}/{score_type}/{L2}_type2_baseline.pt'
            # output_path = f'/home/s2410121/proj_LA/activated_neuron/new_neurons/MMLU-ProX/steer_configs/test.pt'
            torch.save(config, output_path)
            print(f'Saved steer_config to: {output_path}')
    
    """ test """
    # # config = torch.load("/home/s2410121/proj_LA/activated_neuron/new_neurons/MMLU-ProX/steer_configs/steer_config.pt")
    # config = torch.load('/home/s2410121/proj_LA/activated_neuron/new_neurons/MMLU-ProX/steer_configs/aya/cos_sim/ko.pt')

    # for hookpoint, data in config.items():
    #     print(f"Hookpoint: {hookpoint}")
    #     sv = data['steering_vector']
    #     print(f"Steering vector shape: {sv.shape}")
    #     # 特定のニューロンの値を見たいなら
    #     print(f"First 10 values: {sv[0, :15]}")
    #     # ゼロになってるニューロンの位置も確認したい
    #     zero_indices = (sv == 0).nonzero(as_tuple=True)[1]
    #     print(f"Zeroed neuron indices: {zero_indices.tolist()}")
    #     print("------")
    # sys.exit()