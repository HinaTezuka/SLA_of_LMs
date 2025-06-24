from transformers import TrainerCallback
import os
import shutil

class AutoPushCallback(TrainerCallback):
    """
    Automatically push model & tokenizer to HF Hub after each epoch,
    and remove local checkpoints to save disk space.
    """

    def __init__(self, push_repo_name, tokenizer, save_dir="/home/s2410121/proj_LA/activated_neuron/new_neurons/transfer_neurons/model_push_tmp"):
        self.push_repo_name = push_repo_name
        self.tokenizer = tokenizer
        self.save_dir = save_dir

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]

        # Save model & tokenizer to tmp dir
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        print(f"[AutoPushCallback] Saving model to {self.save_dir}")
        model.save_pretrained(self.save_dir)
        self.tokenizer.save_pretrained(self.save_dir)

        # Push to hub
        print(f"[AutoPushCallback] Pushing to Hub: {self.push_repo_name}")
        model.push_to_hub(self.push_repo_name)
        self.tokenizer.push_to_hub(self.push_repo_name)

        # Clean up tmp dir
        shutil.rmtree(self.save_dir)
        print(f"[AutoPushCallback] Cleaned up local save dir: {self.save_dir}")