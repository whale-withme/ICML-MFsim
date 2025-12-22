from transformers import Trainer
"""
让llm接收不仅是target的loss，还有MLP预测的loss
"""
class PolicyTrainer(Trainer):
    def __init__(self, lambda_coeff=0.4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_coeff = lambda_coeff

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1. 解包 inputs
        # 注意：Trainer 会自动把 dataset 的 dict 传给 model.forward(**inputs)
        # 所以 inputs 里已经包含了 current_state, future_states, labels 等
        
        # 2. 前向传播
        outputs = model(**inputs)
        
        # 3. 获取各部分 Loss
        loss_text = outputs["loss"]       # CrossEntropy
        loss_pred = outputs["pred_loss"]  # MSE Sum
        
        # 4. 融合 Loss
        total_loss = (1 - self.lambda_coeff) * loss_text + (self.lambda_coeff * loss_pred)
        
        if self.state.global_step % self.args.logging_steps == 0:
            # 这里的字典 key 就是 TensorBoard 里的曲线名称
            self.log({
                "loss/text_ce": loss_text.detach().item(),
                "loss/pred_kl": loss_pred.detach().item(),
                "loss/total": total_loss.detach().item()
            })
        # 如果需要打印日志，可以在这里记录
        # if self.state.global_step % 10 == 0:
        #     print(f"Step {self.state.global_step}: Text Loss={loss_text.item():.4f}, Pred Loss={loss_pred.item():.4f}")

        return (total_loss, outputs) if return_outputs else total_loss