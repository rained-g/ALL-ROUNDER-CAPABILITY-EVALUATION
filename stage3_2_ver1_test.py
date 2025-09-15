from transformers import AutoTokenizer, LlamaForCausalLM,get_scheduler,pipeline,AutoConfig,AutoModel,AutoImageProcessor
from datasets import load_dataset, concatenate_datasets, Features, Value
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, TaskType, get_peft_model,PeftModel
from torch.optim import AdamW
from collections import namedtuple
from tqdm.notebook import tqdm
from torch.optim.lr_scheduler import LambdaLR
from utils.data_base import *
from utils.tokens_process import ResAny,VisualTokenCompressor
from utils.model_MLP import build_vision_projector
from utils.configset import TrainingConfig,ModelConfig,MultiviewConfig,FusionType
from utils.optimizer_factory import create_train_schdule,create_LoRA_optimizer
from utils.fusion_modules import DynamicRouter,new_DynamicRouter,DynamicRouter1
from utils.obtain_dataloader import AlignDataset_train_all,AlignDataset_test_image_only
from utils.fusing_progress import fuse23,fuse1_4096,fuse23_test,fuse1_4096_test
from functools import partial
from PIL import Image

import torch
import os
import torch.nn as nn
import wandb
import math
import numpy as np
import torch.nn.functional as F
import re
import pandas as pd
import matplotlib.pyplot as plt
import random
from datasets import load_dataset


device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")
print(device1)


#stage-Fuse3
Mcfg = ModelConfig(
    model_save_path= "./results/new-try1/LoRA/Vicuna/your_model-final",
    projetc_save_path="./results/new-try1/MLP/3/Vicuna/your_model.pth",
    target_dim = 4096,
)
Tcfg = TrainingConfig(
    save_dir = "",
    Generate_files = "./results/Generate_files/newtry1/Vicuna/Fuse3_update/ver1",
    epoches=1,
    batch_size=4,
    fusion_type = FusionType.CROSS_ATTN,
    checkpoint_path = './results/new-try1/LoRA/Vicuna/your_model-checkpoints/checkpoint_epoch1_final.pth',
    checkpoint_status = False,
)
Fcfg = MultiviewConfig(
    multia_save_path="./results/new-try1/ATT/3/Vicuna/your_model.pth",
    summary_save_path = ''
)

all_test_new = r'./datasets/MIMIC-complete/processed_data_MARIA2_new/test_data_new.csv'
data_test_avg = r'./datasets/MIMIC-complete/processed_data_MARIA2_new/test_data_avg_43.csv'


#文件内容的读取，使用huggingface的数据读取方式，因为好批量处理
dataset_test_new = load_dataset("csv", data_files=all_test_new, split="train", cache_dir="./datasets_cache/load/all_test_new")
data_test_avg = load_dataset("csv", data_files=data_test_avg, split="train", cache_dir="./datasets_cache/load/data_test_avg_2400")
##分词器以及视觉编码器
LLMs_tokenizer = AutoTokenizer.from_pretrained(Mcfg.vicuna_path)
rad_dino = AutoModel.from_pretrained(Mcfg.dino_path).to(device1)
rad_dino.eval()
processor = AutoImageProcessor.from_pretrained(Mcfg.dino_path)
#Train-model
LLMs_model = LlamaForCausalLM.from_pretrained(Mcfg.vicuna_path,  torch_dtype=torch.bfloat16)
LLMs_model.to(device1)

Tcfg.new_prompt_vicuna()
usual1 = Tcfg.prompt['usual1']
system_prompt = Tcfg.prompt['system_prompt']
P_Re = Tcfg.prompt['P_Re']
user_prompt = Tcfg.prompt['user_prompt']
Tech = Tcfg.prompt['Tech']
Comp = Tcfg.prompt['Comp']
system_prompt_fused1 = Tcfg.prompt['system_prompt_fused1']
system_prompt_fused2 = Tcfg.prompt['system_prompt_fused2']
system_prompt_fused3 = Tcfg.prompt['system_prompt_fused3']
system_prompt_fused = Tcfg.prompt["system_prompt_fused"]



def process_function_stage3_vicuna(examples):
#最终的结果是discrete
    processed_datasets={ }
    #在目标文本部分添加终止符号
    object_i = [i+"}\n" if i else "}\n" for i in examples['indication']]     
    object_pf = [i+"}\n" if i else "}\n" for i in examples['prior_report']]
    object_t = [i+"}\n" if i else "}\n" for i in examples['technique']]
    object_c = [i+"}\n" if i else "}\n" for i in examples['comparison']]
    print(f"indications为{object_i[0]}")
    print(f"prior_findings为{object_pf[0]}")
    print(f"technology为{object_t[0]}")
    print(f"comparisons为{object_c[0]}")
    inputs_prior_text = LLMs_tokenizer(object_pf, padding=True, max_length= 100, truncation=True, return_tensors="pt", add_special_tokens=False)
    inputs_indication = LLMs_tokenizer(object_i, padding=True, max_length= 30, truncation=True, return_tensors="pt", add_special_tokens=False)
    inputs_technology = LLMs_tokenizer(object_t, padding=True, max_length= 15, truncation=True, return_tensors="pt", add_special_tokens=False)
    inputs_comparison = LLMs_tokenizer(object_c, padding=True, max_length= 18, truncation=True, return_tensors="pt", add_special_tokens=False)
    #labels    目标生成文本的input_ids
    processed_datasets['prior'] = inputs_prior_text["input_ids"]
    processed_datasets['prior_att'] = inputs_prior_text['attention_mask']
    processed_datasets['indication'] = inputs_indication["input_ids"]
    processed_datasets['indication_att'] = inputs_indication["attention_mask"]
    processed_datasets['technology'] = inputs_technology["input_ids"]
    processed_datasets['technology_att'] = inputs_technology["attention_mask"]
    processed_datasets['comparison'] = inputs_comparison["input_ids"]
    processed_datasets['comparison_att'] = inputs_comparison["attention_mask"]
    print(f"prior_text的维度为{processed_datasets['prior'].shape},类型为{processed_datasets['prior'].dtype}")
    print(f"indication的维度为{processed_datasets['indication'].shape},类型为{processed_datasets['technology'].dtype}")
    print(f"comparison的维度为{processed_datasets['comparison'].shape},类型为{processed_datasets['comparison'].dtype}")
    print(f"technique的维度为{processed_datasets['technology'].shape},类型为{processed_datasets['technology'].dtype}")
    return processed_datasets

def process_function_vicuna_long(examples):
#最终的结果是discrete
    processed_datasets={ }
    #在目标文本部分添加终止符号
    object_i = [i+"}\n" if i else "}\n" for i in examples['indication']]     
    object_pf = [i+"}\n" if i else "}\n" for i in examples['prior_report']]
    object_t = [i+"}\n" if i else "}\n" for i in examples['technique']]
    object_c = [i+"}\n" if i else "}\n" for i in examples['comparison']]
    print(f"indications为{object_i[0]}")
    print(f"prior_findings为{object_pf[0]}")
    print(f"technology为{object_t[0]}")
    print(f"comparisons为{object_c[0]}")
    inputs_prior_text = LLMs_tokenizer(object_pf, padding=True, max_length= 260, truncation=True, return_tensors="pt", add_special_tokens=False)
    inputs_indication = LLMs_tokenizer(object_i, padding=True, max_length= 80, truncation=True, return_tensors="pt", add_special_tokens=False)
    inputs_technology = LLMs_tokenizer(object_t, padding=True, max_length= 32, truncation=True, return_tensors="pt", add_special_tokens=False)
    inputs_comparison = LLMs_tokenizer(object_c, padding=True, max_length= 32, truncation=True, return_tensors="pt", add_special_tokens=False)
    #labels    目标生成文本的input_ids
    processed_datasets['prior'] = inputs_prior_text["input_ids"]
    processed_datasets['prior_att'] = inputs_prior_text['attention_mask']
    processed_datasets['indication'] = inputs_indication["input_ids"]
    processed_datasets['indication_att'] = inputs_indication["attention_mask"]
    processed_datasets['technology'] = inputs_technology["input_ids"]
    processed_datasets['technology_att'] = inputs_technology["attention_mask"]
    processed_datasets['comparison'] = inputs_comparison["input_ids"]
    processed_datasets['comparison_att'] = inputs_comparison["attention_mask"]
    print(f"prior_text的维度为{processed_datasets['prior'].shape},类型为{processed_datasets['prior'].dtype}")
    print(f"indication的维度为{processed_datasets['indication'].shape},类型为{processed_datasets['technology'].dtype}")
    print(f"comparison的维度为{processed_datasets['comparison'].shape},类型为{processed_datasets['comparison'].dtype}")
    print(f"technique的维度为{processed_datasets['technology'].shape},类型为{processed_datasets['technology'].dtype}")
    return processed_datasets

def process_function_vicuna_thought(examples):
#最终的结果是discrete
    processed_datasets={ }
    #在目标文本部分添加终止符号
    object_i = [i+"}\n" if i else "}\n" for i in examples['indication']]     
    object_pf = [i+"}\n" if i else "}\n" for i in examples['prior_report']]
    object_t = [i+"}\n" if i else "}\n" for i in examples['technique']]
    object_c = [i+"}\n" if i else "}\n" for i in examples['comparison']]
    object_a = [i+"}\n" if i else "}\n" for i in examples['thinking_process_prompt']]
    print(f"indications为{object_i[0]}")
    print(f"prior_findings为{object_pf[0]}")
    print(f"technology为{object_t[0]}")
    print(f"comparisons为{object_c[0]}")
    print(f"analysis为{object_a[0]}")
    inputs_prior_text = LLMs_tokenizer(object_pf, padding=True, max_length= 260, truncation=True, return_tensors="pt", add_special_tokens=False)
    inputs_indication = LLMs_tokenizer(object_i, padding=True, max_length= 80, truncation=True, return_tensors="pt", add_special_tokens=False)
    inputs_technology = LLMs_tokenizer(object_t, padding=True, max_length= 32, truncation=True, return_tensors="pt", add_special_tokens=False)
    inputs_comparison = LLMs_tokenizer(object_c, padding=True, max_length= 32, truncation=True, return_tensors="pt", add_special_tokens=False)
    inputs_ana = LLMs_tokenizer(object_a, padding=True, max_length= 280, truncation=True, return_tensors="pt", add_special_tokens=False)
    #labels    目标生成文本的input_ids
    processed_datasets['prior'] = inputs_prior_text["input_ids"]
    processed_datasets['prior_att'] = inputs_prior_text['attention_mask']
    processed_datasets['indication'] = inputs_indication["input_ids"]
    processed_datasets['indication_att'] = inputs_indication["attention_mask"]
    processed_datasets['technology'] = inputs_technology["input_ids"]
    processed_datasets['technology_att'] = inputs_technology["attention_mask"]
    processed_datasets['comparison'] = inputs_comparison["input_ids"]
    processed_datasets['comparison_att'] = inputs_comparison["attention_mask"]
    processed_datasets['ana'] = inputs_ana["input_ids"]
    processed_datasets['ana_att'] = inputs_ana["attention_mask"]
    print(f"prior_text的维度为{processed_datasets['prior'].shape},类型为{processed_datasets['prior'].dtype}")
    print(f"indication的维度为{processed_datasets['indication'].shape},类型为{processed_datasets['technology'].dtype}")
    print(f"comparison的维度为{processed_datasets['comparison'].shape},类型为{processed_datasets['comparison'].dtype}")
    print(f"technique的维度为{processed_datasets['technology'].shape},类型为{processed_datasets['technology'].dtype}")
    print(f"analysis的维度为{processed_datasets['ana'].shape},类型为{processed_datasets['ana'].dtype}")
    return processed_datasets


processed_all_test_new = dataset_test_new.map(process_function_vicuna_long, batched=True, batch_size=5000, remove_columns=dataset_test_new.column_names, cache_file_name='./datasets_cache/map/Vicuna/all_test_new_stage3_long.arrow')
processed_all_test_new.set_format(type='torch', columns=['prior','prior_att','indication','indication_att','technology','technology_att','comparison','comparison_att'])

processed_test_avg = data_test_avg.map(process_function_vicuna_long, batched=True, batch_size=5000, remove_columns=data_test_avg.column_names, cache_file_name='./datasets_cache/map/Vicuna/data_test_avg_2400_long.arrow')
processed_test_avg.set_format(type='torch', columns=['prior','prior_att','indication','indication_att','technology','technology_att','comparison','comparison_att'])


#从我自己处理过的数据集中的路径处理
real_paths_train_C_F = ['/data1/yyu/mimic-cxr-jpg-2.1.0/files/' + i for i in data_test_avg['Current_frontal_dicom_id']]
real_paths_train_P_F = ['/data1/yyu/mimic-cxr-jpg-2.1.0/files/' + i if pd.notna(i) else None for i in data_test_avg['Prior_frontal_dicom_id']]
real_paths_train_C_L = ['/data1/yyu/mimic-cxr-jpg-2.1.0/files/' + i if pd.notna(i) else None for i in data_test_avg['Current_lateral_dicom_id']]

print(f"训练集的长度为 {len(real_paths_train_C_F)}")
#嵌入转换
usual1_inputs = LLMs_tokenizer(usual1,return_tensors="pt", add_special_tokens=False)
Tcfg.prompt_embed['usual1_embeds'] = LLMs_model.get_input_embeddings()(usual1_inputs['input_ids'].to(LLMs_model.device))
Tcfg.attention_mask['usual1_att'] = usual1_inputs['attention_mask']
del usual1_inputs

system_inputs1 = LLMs_tokenizer(system_prompt_fused1,return_tensors="pt")
Tcfg.prompt_embed['sys1_embeds'] = LLMs_model.get_input_embeddings()(system_inputs1['input_ids'].to(LLMs_model.device))
max_len = Tcfg.prompt_embed['sys1_embeds'].shape[1]
Tcfg.attention_mask['sys1_att'] = system_inputs1['attention_mask']
del system_inputs1

system_inputs = LLMs_tokenizer(system_prompt,return_tensors="pt")
Tcfg.prompt_embed['sys_embeds'] = LLMs_model.get_input_embeddings()(system_inputs['input_ids'].to(LLMs_model.device))
Tcfg.attention_mask['sys_att'] = system_inputs['attention_mask']
# print(Tcfg.attention_mask['sys_att'])
del system_inputs

system_inputs_fused = LLMs_tokenizer(system_prompt_fused,return_tensors="pt")
Tcfg.prompt_embed['sys_embeds_fused'] = LLMs_model.get_input_embeddings()(system_inputs_fused['input_ids'].to(LLMs_model.device))
Tcfg.attention_mask['sys_att_fused'] = system_inputs_fused['attention_mask']
del system_inputs_fused

user_inputs = LLMs_tokenizer(user_prompt,return_tensors="pt", add_special_tokens=False)
Tcfg.prompt_embed['user_embeds'] = LLMs_model.get_input_embeddings()(user_inputs['input_ids'].to(LLMs_model.device))
Tcfg.attention_mask['user_att'] = user_inputs['attention_mask']
del user_inputs

Tech_inputs = LLMs_tokenizer(Tech,return_tensors="pt", add_special_tokens=False)
Tcfg.prompt_embed['Tech_embeds'] = LLMs_model.get_input_embeddings()(Tech_inputs['input_ids'].to(LLMs_model.device))
Tcfg.attention_mask['Tech_att'] = Tech_inputs['attention_mask']
del Tech_inputs

Comp_inputs = LLMs_tokenizer(Comp,return_tensors="pt", add_special_tokens=False)
# Comp_inputs = LLMs_tokenizer(Comp,return_tensors="pt")
Tcfg.prompt_embed['Comp_embeds'] = LLMs_model.get_input_embeddings()(Comp_inputs['input_ids'].to(LLMs_model.device))
Tcfg.attention_mask['Comp_att'] = Comp_inputs['attention_mask']
del Comp_inputs

system_inputs2 = LLMs_tokenizer(system_prompt_fused2,max_length=max_len,padding='max_length',return_tensors="pt")
Tcfg.prompt_embed['sys2_embeds'] = LLMs_model.get_input_embeddings()(system_inputs2['input_ids'].to(LLMs_model.device))
Tcfg.attention_mask['sys2_att'] = system_inputs2['attention_mask']
del system_inputs2

system_inputs3 = LLMs_tokenizer(system_prompt_fused3,max_length=max_len,padding='max_length',return_tensors="pt")
Tcfg.prompt_embed['sys3_embeds'] = LLMs_model.get_input_embeddings()(system_inputs3['input_ids'].to(LLMs_model.device))
Tcfg.attention_mask['sys3_att'] = system_inputs3['attention_mask']
del system_inputs3

P_Re_inputs = LLMs_tokenizer(P_Re,return_tensors="pt", add_special_tokens=False)
Tcfg.prompt_embed['P_Re_embeds'] = LLMs_model.get_input_embeddings()(P_Re_inputs['input_ids'].to(LLMs_model.device))
Tcfg.attention_mask['P_Re_att'] = P_Re_inputs['attention_mask']
del P_Re_inputs

end = LLMs_tokenizer("<s>",return_tensors="pt", add_special_tokens=False)
end_embed = LLMs_model.get_input_embeddings()(end['input_ids'].to(LLMs_model.device))
end_att = end['attention_mask']
#清理空间
torch.cuda.empty_cache()


crossmultiatt = DynamicRouter(view_types=Fcfg.view_types)
if Tcfg.checkpoint_status:
    print("中间参数")
    checkpoint = torch.load(Tcfg.checkpoint_path)
    crossmultiatt.load_state_dict(checkpoint['crossmultiatt'])
else:
    crossmultiatt.load_state_dict(torch.load(Fcfg.multia_save_path))
crossmultiatt = crossmultiatt.to(device0)

class AlignDataset_test(Dataset):
    def __init__(self, data, imgs_CF, imgs_PF, imgs_CL):
        self.imgs_CF = imgs_CF
        self.imgs_PF = imgs_PF
        self.imgs_CL = imgs_CL
        self.data = data

    def __len__(self):
        return len(self.imgs_CF)

    def __getitem__(self, idx):
        flag_pf = 0
        flag_cl = 0
        # 图像嵌入
        views = {}
        #得到CF_patches图像
        full_image = Image.open(self.imgs_CF[idx]).convert("RGB")
        CF_patches, M,N = ResAny(full_image,5)
        #CF
        inputs = processor(images=full_image, return_tensors="pt").to(device1)
        with torch.inference_mode():
            outputs = rad_dino(**inputs)
            features = outputs.last_hidden_state.squeeze(0)   #[1370,768]
        views['cf'] = features[1:,:].to(next(crossmultiatt.parameters()).device)
        #PF
        if self.imgs_PF[idx] :
            flag_pf = 1
            full_image = Image.open(self.imgs_PF[idx]).convert("RGB")
            inputs = processor(images=full_image, return_tensors="pt").to(device1)
            with torch.inference_mode():
                outputs = rad_dino(**inputs)
                features = outputs.last_hidden_state.squeeze(0)   #[1370,768]
            views['pf'] = features[1:,:].to(next(crossmultiatt.parameters()).device)
        if self.imgs_CL[idx] :
            flag_cl = 1
            full_image = Image.open(self.imgs_CL[idx]).convert("RGB")
            inputs = processor(images=full_image, return_tensors="pt").to(device1)
            with torch.inference_mode():
                outputs = rad_dino(**inputs)
                features = outputs.last_hidden_state.squeeze(0)   #[1370,768]
            views['cl'] = features[1:,:].to(next(crossmultiatt.parameters()).device)

        image_feature = crossmultiatt(features_dict=views)

        del views
        torch.cuda.empty_cache()

        if flag_cl and flag_pf:
            input_type = 'CF + CL + PF'
        elif flag_pf:
            input_type = 'CF + PF'
        elif flag_cl:
            input_type = 'CF + CL'
        else:
            input_type = 'CF only'
        #文本嵌入
        #indication_embed
        ind_embeds = LLMs_model.get_input_embeddings()(self.data['indication'][idx].to(LLMs_model.device))
        #technique_embed
        tech_embeds = LLMs_model.get_input_embeddings()(self.data['technology'][idx].to(LLMs_model.device))
        #comparison_embeds
        comp_embeds = LLMs_model.get_input_embeddings()(self.data['comparison'][idx].to(LLMs_model.device))
        #prior_f
        prior_f_embeds = LLMs_model.get_input_embeddings()(self.data['prior'][idx].to(LLMs_model.device))


        return image_feature, self.data['indication_att'][idx],ind_embeds, self.data['technology_att'][idx],tech_embeds, self.data['comparison_att'][idx],comp_embeds, self.data['prior_att'][idx],prior_f_embeds

class AlignDataset_test_all(Dataset):
    def __init__(self, data, imgs_CF, imgs_PF, imgs_CL):
        self.imgs_CF = imgs_CF
        self.imgs_PF = imgs_PF
        self.imgs_CL = imgs_CL
        self.data = data
        print(len(self.data))

    def __len__(self):
        return len(self.imgs_CF)

    def __getitem__(self, idx):
        #文本嵌入
        #indication_embed
        ind_embeds = LLMs_model.get_input_embeddings()(self.data['indication'][idx])
        #technique_embed
        tech_embeds = LLMs_model.get_input_embeddings()(self.data['technology'][idx])
        #comparison_embeds
        comp_embeds = LLMs_model.get_input_embeddings()(self.data['comparison'][idx])
        #prior_f
        prior_f_embeds = LLMs_model.get_input_embeddings()(self.data['prior'][idx])
        # 图像嵌入
        #1
        if self.imgs_CF[idx] is None:
            img_CF_feature = "None"
        else:
            # print(f"CF路径：{self.imgs_CF[idx]}")
            full_image = Image.open(self.imgs_CF[idx]).convert("RGB")
            inputs = processor(images=full_image, return_tensors="pt").to(device1)
            with torch.inference_mode():
                outputs = rad_dino(**inputs)
                features = outputs.last_hidden_state.squeeze(0)   #[1370,768]
            img_CF_feature = features[1:,:].to(device0)
        #2
        if self.imgs_PF[idx] is None:
            img_PF_feature = "None"
        else:
            full_image = Image.open(self.imgs_PF[idx]).convert("RGB")
            inputs = processor(images=full_image, return_tensors="pt").to(device1)
            with torch.inference_mode():
                outputs = rad_dino(**inputs)
                features = outputs.last_hidden_state.squeeze(0)   #[1370,768]
            img_PF_feature = features[1:,:].to(device0)
        #3
        if self.imgs_CL[idx] is None:
            img_CL_feature = "None"
        else:
            full_image = Image.open(self.imgs_CL[idx]).convert("RGB")
            inputs = processor(images=full_image, return_tensors="pt").to(device1)
            with torch.inference_mode():
                outputs = rad_dino(**inputs)
                features = outputs.last_hidden_state.squeeze(0)   #[1370,768]
            img_CL_feature = features[1:,:].to(device0)
        #扩充labels(好像不能再这里扩充labels，因为后面会对输入进行缩放，要等缩放之后在进行),直接在这里缩放试试,不行跑这个要两张显存80G的卡
        del features
        return img_CF_feature, img_PF_feature, img_CL_feature, self.data['indication_att'][idx],ind_embeds, self.data['technology_att'][idx],tech_embeds, self.data['comparison_att'][idx],comp_embeds, self.data['prior_att'][idx],prior_f_embeds
    
dataset_test = AlignDataset_test(processed_test_avg,real_paths_train_C_F,real_paths_train_P_F,real_paths_train_C_L)
dataloader_test = DataLoader(dataset_test, batch_size=Tcfg.batch_size, shuffle=False)

Config = namedtuple('Config', ['mm_projector_type', 'mm_hidden_size', 'hidden_size',"target_dim"])
# 创建一个 Config 实例
config = Config(mm_projector_type=Mcfg.projector_type, mm_hidden_size=Mcfg.mm_hidden_size, hidden_size=Mcfg.hidden_size, target_dim=Mcfg.target_dim)
# 创建视觉投影器
projector = build_vision_projector(config,Mcfg)
if Tcfg.checkpoint_status:
    peft_LLMs = PeftModel.from_pretrained(LLMs_model, checkpoint["peft_LLMs_lora_path"])
    projector.load_state_dict(checkpoint['projector_state_dict'])
else:
    peft_LLMs = PeftModel.from_pretrained(LLMs_model, Mcfg.model_save_path)
    projector.load_state_dict(torch.load(Mcfg.projetc_save_path))
projector.to(peft_LLMs.device)

projector.eval()
peft_LLMs.eval()
crossmultiatt.eval()

projector_device = next(projector.parameters()).device


for i in range(1):
    save_path = Tcfg.Generate_files
    if Tcfg.fusion_type == FusionType.CROSS_ATTN_NEW:
        file_name = f"Vicuna_NewFuse3_stage3_3w_att_standard{i+1}.csv"
    else:
        file_name = f"Vicuna_your_model_att_avg_256_standard{i+1}.csv"
    os.makedirs(save_path, exist_ok=True)
    progress_bar_train = tqdm(dataloader_test, leave=True, mininterval=1.0)
    results = []
    for step, batch in enumerate(progress_bar_train):
        image,ind_att,ind,tech_att,tech,comp_att,comp,pf_att,pf = batch
        # 获取当前批次的实际大小
        current_batch_size = ind.shape[0]
        #视觉转化
        vision_embeds = projector(image.to(projector_device))
        vision_len = vision_embeds.shape[1]
        vision_mask = torch.ones(current_batch_size, vision_len, dtype=torch.long, device=Tcfg.attention_mask['P_Re_att'].device)
        #扩展提示嵌入以匹配视觉特征
        expand_sys_embeds = Tcfg.prompt_embed['sys_embeds_fused'].expand(current_batch_size,-1,-1)
        expand_user_embeds = Tcfg.prompt_embed['user_embeds'].expand(current_batch_size,-1,-1)
        expand_usual1_embeds = Tcfg.prompt_embed['usual1_embeds'].expand(current_batch_size,-1,-1)
        expand_pf_embeds = Tcfg.prompt_embed['P_Re_embeds'].expand(current_batch_size,-1,-1)
        expand_Tech_embeds = Tcfg.prompt_embed['Tech_embeds'].expand(current_batch_size,-1,-1)
        expand_Comp_embeds = Tcfg.prompt_embed['Comp_embeds'].expand(current_batch_size,-1,-1)
        expand_end_embeds = end_embed.expand(current_batch_size,-1,-1)

        expand_sys_att = Tcfg.attention_mask['sys_att_fused'].expand(current_batch_size,-1)
        expand_usual1_att = Tcfg.attention_mask['usual1_att'].expand(current_batch_size,-1)
        expand_user_att = Tcfg.attention_mask['user_att'].expand(current_batch_size,-1)
        expand_pf_att = Tcfg.attention_mask['P_Re_att'].expand(current_batch_size,-1)
        expand_Tech_att = Tcfg.attention_mask['Tech_att'].expand(current_batch_size,-1)
        expand_Comp_att = Tcfg.attention_mask['Comp_att'].expand(current_batch_size,-1)
        expand_end_att = end_att.expand(current_batch_size,-1)
        #结合文本提示以及视觉嵌入
        input_embeddings = torch.cat([expand_sys_embeds,vision_embeds,expand_usual1_embeds],dim=1)
        # print(expand_sys_att.shape,vision_mask.shape,expand_usual1_att.shape)
        att_mask = torch.cat([expand_sys_att, vision_mask, expand_usual1_att], dim=-1)
        del expand_sys_embeds,vision_embeds

        torch.cuda.empty_cache()
        #结合pf
        input_embeddings = torch.cat([input_embeddings,expand_pf_embeds,pf],dim=1)
        del expand_pf_embeds,pf
        #结合用户提示以及indication
        input_embeddings = torch.cat([input_embeddings,expand_user_embeds,ind],dim=1)
        del expand_user_embeds, ind
        #结合technique
        input_embeddings = torch.cat([input_embeddings,expand_Tech_embeds,tech],dim=1)
        del expand_Tech_embeds, tech
        components_att = [
            att_mask,
            expand_pf_att,
            pf_att,
            expand_user_att,
            ind_att,
            expand_Tech_att,
            tech_att,
        ]

        #将ttention进行改变
        att_mask = torch.cat(components_att,dim=-1).to(peft_LLMs.device)
        del components_att

        torch.cuda.empty_cache() 
        print(f"最终的输入嵌入形状为{input_embeddings.shape},最终输入的注意力形状为{att_mask.shape}")
        with torch.no_grad():
            outputs = peft_LLMs.generate(inputs_embeds=input_embeddings.to(torch.bfloat16),
                                attention_mask = att_mask,
                                max_new_tokens = 256,
                                min_new_tokens = 60, 
                                repetition_penalty=1.2,     # 加大重复惩罚
                                do_sample=True,             # 启用采样
                                temperature=0.9,            # 设置温度
                                top_k=50,                   # top-k采样
                                top_p=0.6,                  # top-p采样
                                )
            temp = [LLMs_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            for i in range(len(temp)):
                results.append(temp[i])
                print(temp[i])
            print(len(results))
    df = pd.DataFrame({"Report Impression": results})
    df['study_id'] = data_test_avg['study_id']
    full_save_path = os.path.join(save_path, file_name)
    df.to_csv(full_save_path, index=False)

    print(f"数据已成功写入 {full_save_path} 文件。")

