# optimizer_factory.py
from utils.configset import TrainingConfig, FusionType
from torch.optim import AdamW
import torch
import math

def create_optimizer(config: TrainingConfig, peft_model, projector, multiatt):
    params = [
        {"params": projector.parameters(), "lr": config.lr_settings["projector"]},
        {"params": peft_model.parameters(), "lr": config.lr_settings["lora"]}

    ]
    
    if config.fusion_type in [FusionType.SELF_ATTN, FusionType.CROSS_ATTN]:
        params.append({"params": multiatt.parameters(), "lr": config.lr_settings["fusion"]})
    
    return AdamW(params, weight_decay=0.01)

def clip_gradients(config: TrainingConfig, peft_model, projector, multiatt):
    params_config = [
        {'params': projector.parameters(), 'max_norm': 5.0},
        {'params': peft_model.parameters(), 'max_norm': 1.0},     
    ]

    if config.fusion_type in [FusionType.SELF_ATTN, FusionType.CROSS_ATTN]:
        params_config.append({"params": multiatt.parameters(), "max_norm": 3.0})

    for i in params_config:
        torch.nn.utils.clip_grad_norm_(
            i['params'], 
            i['max_norm'],
            norm_type=2
        )


#_____________________________________________________________________________________________下面适用于非模块化
def create_most_optimizer(projector, peft_model, multiatt=None):
    params = [
        {"params": projector.parameters(), "lr": 5e-5},
        {"params": peft_model.parameters(), "lr": 2e-5}
    ]
    
    if multiatt!=None:
        params.append({"params": multiatt.parameters(), "lr": 3e-5})
    
    return AdamW(params, weight_decay=0.01)

def create_LoRA_optimizer(rate, projector, peft_model, summary=None, multiatt=None, view=None):
    params = [
        {"params": projector.parameters(), "lr": 5e-5 * math.sqrt(rate)},          #4e-5
        {"params": peft_model.parameters(), "lr": 2e-5 * math.sqrt(rate)}           #1.5e-4
    ]

    
    if summary!=None:
        params.append({"params": summary.parameters(), "lr": 5e-5 * math.sqrt(rate)})             #4e-4

    if multiatt!=None:
        params.append({"params": multiatt.parameters(), "lr": 3e-5 * math.sqrt(rate)})            #5e-4    

    if view!=None:
        params.append({"params": view.parameters(), "lr": 5e-5 * math.sqrt(rate)})            #5e-4   

    return AdamW(params, weight_decay=0.01)

def create_stage3_optimizer(rate, projector, peft_model, summary=None, multiatt=None, view=None):
    params = [
        {"params": projector.parameters(), "lr": 5e-5 },          #4e-5
        {"params": peft_model.parameters(), "lr": 2e-5 * math.sqrt(rate)}           #1.5e-4
    ]

    
    if summary!=None:
        params.append({"params": summary.parameters(), "lr": 5e-5 * math.sqrt(rate)})             #4e-4

    if multiatt!=None:
        params.append({"params": multiatt.parameters(), "lr": 3e-5 * math.sqrt(rate)})            #5e-4    

    if view!=None:
        params.append({"params": view.parameters(), "lr": 5e-5 * math.sqrt(rate)})            #5e-4   

    return AdamW(params, weight_decay=0.01)

def create_stage1_optimizer(rate, projector, peft_model, multiatt=None):
    params = [
        {"params": projector.parameters(), "lr": 3e-5 },          #4e-5
        {"params": peft_model.parameters(), "lr": 2e-5 * math.sqrt(rate)}           #1.5e-4
    ]
    if multiatt!=None:
        params.append({"params": multiatt.parameters(), "lr": 2.5e-5 * math.sqrt(rate)})            #5e-4    

    return AdamW(params, weight_decay=0.01)

# def create_stage3_optimizer(projector, peft_model, summary=None, multiatt=None,stage=0):
#     params = [
#         {"params": projector.parameters(), "lr": 4e-5},          #4e-5
#         {"params": peft_model.parameters(), "lr": 1e-4}           #1.5e-4
#     ]

    
#     if summary!=None:
#         params.append({"params": summary.parameters(), "lr": 1e-4})             #4e-4

#     if multiatt!=None:
#         params.append({"params": multiatt.parameters(), "lr": 5e-4})          

#     return AdamW(params, weight_decay=0.01)


def create_most_schdule(current_step: int, num_training_steps,num_warmup_steps,num):
    if current_step < num_warmup_steps:
        return [current_step / num_warmup_steps for _ in range(num)]
    # 计算进度比例
    progress = (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
    print(f"{progress*100:.2f}%")
    # 阶段判断
    if progress <= 0.3:  
        # print("2")
        return [1.0 for _ in range(num)]
    
    elif progress <= 0.7:  # 中期阶段（30-70%）
        scale_factors = [
            0.6,    # projector衰减到初始的60%
            1.0,    # lora保持
            1.0     # crossmultiatt保持
        ]
        return scale_factors
    
    else:  # 后期阶段（70-100%）
        decay_factor = max(0.1, 0.1 * (1 - progress))  # 线性衰减到初始的10%
        return [decay_factor for _ in range(num)]

def create_DL_optimizer(projector, peft_model, multiatt=None):
    params = [
        {"params": projector.parameters(), "lr": 5e-4},
        {"params": peft_model.parameters(), "lr": 2e-4}
    ]
    
    if multiatt!=None:
        params.append({"params": multiatt.parameters(), "lr": 3e-4})
    
    return AdamW(params, weight_decay=0.01) 

def create_train_linear_schdule(current_step: int, num_training_steps,num_warmup_steps):
    # Warmup阶段（前10%）
    if current_step < num_warmup_steps:
        warmup_ratio = current_step / num_warmup_steps
        return [
            warmup_ratio**0.5,  # projector快速上升
            warmup_ratio**1.5,   # peft慢启动
            warmup_ratio          # crossmultiatt线性
        ]
    
    # 主训练阶段（统一余弦退火）
    progress = (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    
    return [
        max(0.3, 0.7 + 0.3*cosine_decay),  # projector: 70%→30%
        0.5 + 0.5*cosine_decay,            # peft: 100%→50%
        max(0.6, 0.8 + 0.2*cosine_decay)    # crossmultiatt: 80%→60%
    ]

def create_train_schdule(current_step: int, num_training_steps,num_warmup_steps):
    # Warmup阶段（前10%）
    if current_step < num_warmup_steps:
        warmup_ratio = current_step / num_warmup_steps
        return [
            warmup_ratio**0.5,  # projector快速上升
            warmup_ratio**1.5,   # peft慢启动
            warmup_ratio,           # summmary线性
            warmup_ratio,          # crossmultiatt线性
            warmup_ratio**0.5
        ]
    
    # 主训练阶段（统一余弦退火）
    progress = (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    
    return [
        max(0.3, 0.7 + 0.3*cosine_decay),  # projector: 70%→30%
        0.5 + 0.5*cosine_decay,            # peft: 100%→50%
        max(0.4, 0.6 + 0.4*cosine_decay),    # summary: 100%→40%
        max(0.6, 0.8 + 0.2*cosine_decay),    # crossmultiatt: 80%→60%
        max(0.3, 0.7 + 0.3*cosine_decay)
    ]
    
def create_train_schedule(current_step, num_training_steps, num_warmup_steps, phase: str):
    # Warmup ratio
    if current_step < num_warmup_steps:
        warmup_ratio = current_step / num_warmup_steps
        if phase == 'phase1':
            return [warmup_ratio**0.5, warmup_ratio**1.5, warmup_ratio]
        if phase == 'phase2':
            return [warmup_ratio, warmup_ratio**1.5, warmup_ratio]
        if phase == 'phase3':
            return [warmup_ratio**0.3, warmup_ratio**2, warmup_ratio**0.5]
    
    # Cosine decay
    progress = (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

    if phase == 'phase1':
        return [
            max(0.4, 0.8 * cosine_decay),  # projector
            0.2 + 0.6 * cosine_decay,      # peft
            max(0.5, 0.7 * cosine_decay),  # crossmultiatt
        ]
    elif phase == 'phase2':
        return [
            max(0.3, 0.6 * cosine_decay),  # projector
            0.5 + 0.5 * cosine_decay,      # peft
            0.6 + 0.3 * cosine_decay,      # crossmultiatt
        ]
    elif phase == 'phase3':
        return [
            0.2,                           # projector frozen
            0.6 + 0.4 * cosine_decay,      # peft full train
            0.4,                           # crossmultiatt low lr
        ]
    '''
# 定义学习率调度器
# 保存各组的初始学习率
# initial_lrs = [group['lr'] for group in optimizer.param_groups]
# num_training_steps = len(dataloader_train) * epochs  
# num_warmup_steps = int(0.15 * num_training_steps)  # 预热步数（占总步数的 15%）

grad_accum_steps = 16

def create_train_schdule(current_step: int, num_training_steps,num_warmup_steps):
    # Warmup阶段（前10%）
    if current_step < num_warmup_steps:
        warmup_ratio = current_step / num_warmup_steps
        return [
            warmup_ratio**0.5,  # projector快速上升
            warmup_ratio**1.5,   # peft慢启动
            warmup_ratio          # crossmultiatt线性
        ]
    
    # 主训练阶段（统一余弦退火）
    progress = (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    
    return [
        max(0.3, 0.7 + 0.3*cosine_decay),  # projector: 70%→30%
        0.5 + 0.5*cosine_decay,            # peft: 100%→50%
        max(0.6, 0.8 + 0.2*cosine_decay)    # crossmultiatt: 80%→60%
    ]

# 创建调度器
scheduler = LambdaLR(
    optimizer,
    lr_lambda=[lambda step: custom_schedule(step)[i] for i in range(len([group['lr'] for group in optimizer.param_groups]))]
)
# 梯度裁剪（分模块差异化设置）
def clip_gradients():
    params_config = [
        {'params': projector.parameters(), 'max_norm': 2.5},  # 原3.0→2.5
        {'params': crossmultiatt.parameters(), 'max_norm': 1.8},  # 原2.0→1.8
        {'params': peft_vicuna.parameters(), 'max_norm': 1.5},  # 原2.0→1.5
    ]
    # 添加梯度累积补偿因子
    scale = 1.0 / math.sqrt(grad_accum_steps)
    for config in params_config:
        torch.nn.utils.clip_grad_norm_(
            config['params'], 
            config['max_norm'] * scale,  # 动态调整阈值
            norm_type=2
        )
        '''


'''
old
optimizer = AdamW([
    {"params": projector.parameters(), "lr": 5e-5},  # 对齐小数据集最佳参数
    {"params": peft_vicuna.parameters(), "lr": 2e-5},  # 严格遵循MARIA2基准
    {"params": crossmultiatt.parameters(), "lr": 3e-5},  # 折中参数
], weight_decay=0.01)
epochs = 1
# 定义学习率调度器
# 保存各组的初始学习率
initial_lrs = [group['lr'] for group in optimizer.param_groups]
num_training_steps = len(dataloader_train) * epochs  
num_warmup_steps = int(0.03 * num_training_steps)  # 预热步数（占总步数的 15%）
def custom_schedule(current_step: int,warmup,total_step):

    # Warmup阶段（前5%）
    if current_step < num_warmup_steps:
        # print(1)
        
        return [current_step / num_warmup_steps for _ in initial_lrs]
    
    # 计算进度比例
    progress = (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
    print(f"{progress*100:.2f}%")
    # 阶段判断
    if progress <= 0.3:  
        # print("2")
        return [1.0 for _ in initial_lrs]
    
    elif progress <= 0.7:  # 中期阶段（30-70%）
        scale_factors = [
            0.6,    # projector衰减到初始的60%
            1.0,    # lora保持
            1.0     # crossmultiatt保持
        ]
 
        return scale_factors
    
    else:  # 后期阶段（70-100%）
        decay_factor = max(0.1, 0.1 * (1 - progress))  # 线性衰减到初始的10%
        return [decay_factor for _ in initial_lrs]

# 创建调度器
scheduler = LambdaLR(
    optimizer,
    lr_lambda=[lambda step: custom_schedule(step)[i] for i in range(len(initial_lrs))]
)
def clip_gradients():
    params_config = [
        {'params': projector.parameters(), 'max_norm': 5.0},
        {'params': crossmultiatt.parameters(), 'max_norm': 3.0},
        {'params': peft_vicuna.parameters(), 'max_norm': 2.0},
        
    ]
    
    for config in params_config:
        torch.nn.utils.clip_grad_norm_(
            config['params'], 
            config['max_norm'],
            norm_type=2
        )
'''