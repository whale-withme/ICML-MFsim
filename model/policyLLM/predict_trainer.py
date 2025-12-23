import os
import torch
from transformers import Trainer, TrainerCallback

# 回调函数：确保 checkpoint 文件夹里包含 MLP 权重，用于断点续训
class SaveMLPCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        
        model = kwargs['model']
        # 处理可能存在的分布式包装
        model_to_save = model.module if hasattr(model, 'module') else model
        
        torch.save(
            model_to_save.prediction_heads.state_dict(), 
            os.path.join(checkpoint_path, "prediction_heads.pt")
        )

class PolicyTrainer(Trainer):
    def __init__(self, lambda_coeff=0.4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_coeff = lambda_coeff

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss_text = outputs["loss"]
        loss_pred = outputs["pred_loss"]
        
        total_loss = (1 - self.lambda_coeff) * loss_text + (self.lambda_coeff * loss_pred)
        
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "loss/text_ce": loss_text.detach().item(),
                "loss/pred_kl": loss_pred.detach().item(),
                "loss/total": total_loss.detach().item()
            })

        return (total_loss, outputs) if return_outputs else total_loss