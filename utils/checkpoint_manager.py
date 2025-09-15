# checkpoint_manager.py
import os
import torch
from utils.configset import TrainingConfig, FusionType

class CheckpointManager:
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        os.makedirs(config.save_dir, exist_ok=True)
        
    def save(self, epoch,step, projector, optimizer, scheduler, path, fusion_module=None):
        # 自动生成保存路径
        checkpoint_path = os.path.join(self.config.save_dir, f"checkpoint_epoch{epoch+1}_step{step}.pth")
        # 通用组件
        checkpoint = {
                "projector_state_dict": projector.state_dict(),
                "peft_vicuna_lora_path": path,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
        # 融合模块特殊处理
        if self.config.fusion_type != FusionType.LINEAR:
            checkpoint["fusion_module"] = fusion_module.state_dict()
        
        # 保存
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at: {checkpoint_path}")

# 主程序中使用
# from checkpoint_manager import CheckpointManager

# saver = CheckpointManager(cfg)
# saver.save(step, model, peft_vicuna, optimizer)