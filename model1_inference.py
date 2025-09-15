from transformers import AutoTokenizer, LlamaForCausalLM,AutoImageProcessor,pipeline,AutoModel
from datasets import load_dataset, concatenate_datasets
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, TaskType, get_peft_model,PeftModel
from torch.optim import AdamW
from collections import namedtuple
from tqdm.notebook import tqdm
from torch.optim.lr_scheduler import LambdaLR
from utils.data_base import *
from utils.model_MLP import build_vision_projector
from utils.data_base import get_path
from utils.configset import TrainingConfig, FusionType,ModelConfig,MultiviewConfig
from utils.optimizer_factory import create_most_optimizer,create_most_schdule,create_train_schdule
from utils.obtain_dataloader import AlignDataset_test_all,AlignDataset_test_image_only
from utils.fusing_progress import fuse4_test,fuse1_4096_test,fuse23_test
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
import random


device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")
print(device1)

# %%
Mcfg = ModelConfig(
    model_save_path= "./results/new-try1/LoRA/Vicuna/small_maira2_new_3W2-final",
    projetc_save_path="./results/new-try1/MLP/1/Vicuna/small_maira2_new_3W2.pth",
    target_dim = 4096,
)
Tcfg = TrainingConfig(
    save_dir = "",
    Generate_files = "./results/Generate_files/newtry1/Vicuna/small_maira2",
    epoches=1,
    batch_size=1,
    checkpoint_path = './results/new-try1/LoRA/Vicuna/small_maira2_new_3W2-checkpoints/checkpoint_epoch3_step25716.pth',
    checkpoint_status = True,
    fusion_type = FusionType.LINEAR,
)
Fcfg = MultiviewConfig(
    multia_save_path="",
)

all_test_new = r'./datasets/MIMIC-complete/processed_data_MARIA2_new/test_data_new.csv'
data_test_avg = r'./datasets/MIMIC-complete/processed_data_MARIA2_new/test_data_avg_43.csv'
#文件内容的读取，使用huggingface的数据读取方式，因为好批量处理
dataset_test_new = load_dataset("csv", data_files=all_test_new, split="train", cache_dir="./datasets_cache/load/all_test_new")
data_test_avg = load_dataset("csv", data_files=data_test_avg, split="train", cache_dir="./datasets_cache/load/data_test_avg_2400")

##分词器以及视觉编码器
LLMs_tokenizer = AutoTokenizer.from_pretrained(Mcfg.vicuna_path)
# pipe_image = pipeline(task="image-feature-extraction", model=Mcfg.dino_path, pool=False, device=1)
LLMs_model = LlamaForCausalLM.from_pretrained(Mcfg.vicuna_path,  torch_dtype=torch.bfloat16)
LLMs_model.to(device0)
rad_dino = AutoModel.from_pretrained(Mcfg.dino_path).to(LLMs_model.device)
rad_dino.eval()
processor = AutoImageProcessor.from_pretrained(Mcfg.dino_path)
#指定对应的系统指示以及用户指示
Tcfg.new_prompt_vicuna_maira2()
usual1 = Tcfg.prompt['usual1']
system_prompt = Tcfg.prompt['system_prompt']
C_lateral = Tcfg.prompt['C_lateral']
P_frontal = Tcfg.prompt['P_frontal']
patch_prompt = Tcfg.prompt['patches_image']
P_Re = Tcfg.prompt['P_Re']
user_prompt = Tcfg.prompt['user_prompt']
Tech = Tcfg.prompt['Tech']
Comp = Tcfg.prompt['Comp']
system_prompt_fused = Tcfg.prompt["system_prompt_fused"]

# %%
def process_function_vicuna(examples):
#最终的结果是discrete
    processed_datasets={ }
    #在目标文本部分添加终止符号
    object_i = [i+"}\n" if i else "}\n" for i in examples['indication']]     
    object_pf = [i+"}"+ "</s>" if i else "}</s>" for i in examples['prior_report']]
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

# %%
processed_all_test_new = dataset_test_new.map(process_function_vicuna, batched=True, batch_size=1000, remove_columns=dataset_test_new.column_names, cache_file_name='./datasets_cache/map/Vicuna/all_test_new_stage3.arrow')
processed_all_test_new.set_format(type='torch', columns=['prior','prior_att','indication','indication_att','technology','technology_att','comparison','comparison_att'])

processed_test_avg = data_test_avg.map(process_function_vicuna, batched=True, batch_size=5000, remove_columns=data_test_avg.column_names, cache_file_name='./datasets_cache/map/Vicuna/data_test_avg_2400.arrow')
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

system_inputs = LLMs_tokenizer(system_prompt,return_tensors="pt")
Tcfg.prompt_embed['sys_embeds'] = LLMs_model.get_input_embeddings()(system_inputs['input_ids'].to(LLMs_model.device))
Tcfg.attention_mask['sys_att'] = system_inputs['attention_mask']
del system_inputs

C_lateral_inputs = LLMs_tokenizer(C_lateral,return_tensors="pt", add_special_tokens=False)
Tcfg.prompt_embed['C_lateral_embed'] = LLMs_model.get_input_embeddings()(C_lateral_inputs['input_ids'].to(LLMs_model.device))
Tcfg.attention_mask['CL_att'] = C_lateral_inputs['attention_mask']
del C_lateral_inputs

P_frontal_inputs = LLMs_tokenizer(P_frontal,return_tensors="pt", add_special_tokens=False)
Tcfg.prompt_embed['P_frontal_embeds'] = LLMs_model.get_input_embeddings()(P_frontal_inputs['input_ids'].to(LLMs_model.device))
Tcfg.attention_mask['PF_att'] = P_frontal_inputs['attention_mask']
del P_frontal_inputs

P_Re_inputs = LLMs_tokenizer(P_Re,return_tensors="pt", add_special_tokens=False)
Tcfg.prompt_embed['P_Re_embeds'] = LLMs_model.get_input_embeddings()(P_Re_inputs['input_ids'].to(LLMs_model.device))
Tcfg.attention_mask['P_Re_att'] = P_Re_inputs['attention_mask']
del P_Re_inputs

user_inputs = LLMs_tokenizer(user_prompt,return_tensors="pt")
Tcfg.prompt_embed['user_embeds'] = LLMs_model.get_input_embeddings()(user_inputs['input_ids'].to(LLMs_model.device))
Tcfg.attention_mask['user_att'] = user_inputs['attention_mask']
del user_inputs

Tech_inputs = LLMs_tokenizer(Tech,return_tensors="pt", add_special_tokens=False)
Tcfg.prompt_embed['Tech_embeds'] = LLMs_model.get_input_embeddings()(Tech_inputs['input_ids'].to(LLMs_model.device))
Tcfg.attention_mask['Tech_att'] = Tech_inputs['attention_mask']
del Tech_inputs

Comp_inputs = LLMs_tokenizer(Comp,return_tensors="pt", add_special_tokens=False)
Tcfg.prompt_embed['Comp_embeds'] = LLMs_model.get_input_embeddings()(Comp_inputs['input_ids'].to(LLMs_model.device))
Tcfg.attention_mask['Comp_att'] = Comp_inputs['attention_mask']
del Comp_inputs

#清理空间
torch.cuda.empty_cache()

class AlignDataset_Fuse1_test(Dataset):
    def __init__(self, data, imgs_CF, imgs_PF, imgs_CL):
        self.imgs_CF = imgs_CF
        self.imgs_PF = imgs_PF
        self.imgs_CL = imgs_CL
        self.data = data

    def __len__(self):
        return len(self.imgs_CF)

    def __getitem__(self, idx):
        # 图像嵌入
        #得到CF_patches图像
        full_image = Image.open(self.imgs_CF[idx]).convert("RGB")
        #CF
        inputs = processor(images=full_image, return_tensors="pt").to(rad_dino.device)
        with torch.inference_mode():
            outputs = rad_dino(**inputs)
            features = outputs.last_hidden_state.squeeze(0)   #[1370,768]
        img_CF_feature = features[1:,:]
        #PF
        if self.imgs_PF[idx] is not None:
            full_image = Image.open(self.imgs_PF[idx]).convert("RGB")
            inputs = processor(images=full_image, return_tensors="pt").to(rad_dino.device)
            with torch.inference_mode():
                outputs = rad_dino(**inputs)
                features = outputs.last_hidden_state.squeeze(0)   #[1370,768]
            img_PF_feature = features[1:,:]
        else:
            img_PF_feature = "None"

        if self.imgs_CL[idx] is not None:
            full_image = Image.open(self.imgs_CL[idx]).convert("RGB")
            inputs = processor(images=full_image, return_tensors="pt").to(rad_dino.device)
            with torch.inference_mode():
                outputs = rad_dino(**inputs)
                features = outputs.last_hidden_state.squeeze(0)   #[1370,768]
            img_CL_feature = features[1:,:]
        else:
            img_CL_feature = "None"
        torch.cuda.empty_cache()
        #文本嵌入
        #indication_embed
        ind_embeds = LLMs_model.get_input_embeddings()(self.data['indication'][idx].to(LLMs_model.device))
        #technique_embed
        tech_embeds = LLMs_model.get_input_embeddings()(self.data['technology'][idx].to(LLMs_model.device))
        #comparison_embeds
        comp_embeds = LLMs_model.get_input_embeddings()(self.data['comparison'][idx].to(LLMs_model.device))
        #prior_f
        prior_f_embeds = LLMs_model.get_input_embeddings()(self.data['prior'][idx].to(LLMs_model.device))


        return img_CF_feature, img_PF_feature, img_CL_feature,  self.data['indication_att'][idx],ind_embeds, self.data['technology_att'][idx],tech_embeds, self.data['comparison_att'][idx],comp_embeds, self.data['prior_att'][idx],prior_f_embeds


dataset_test = AlignDataset_Fuse1_test(processed_test_avg,real_paths_train_C_F,real_paths_train_P_F,real_paths_train_C_L)
dataloader_test = DataLoader(dataset_test, batch_size=Tcfg.batch_size, shuffle=False)

Config = namedtuple('Config', ['mm_projector_type', 'mm_hidden_size', 'hidden_size',"target_dim"])
# 创建一个 Config 实例
config = Config(mm_projector_type=Mcfg.projector_type, mm_hidden_size=Mcfg.mm_hidden_size, hidden_size=Mcfg.hidden_size, target_dim=Mcfg.target_dim)
# 创建视觉投影器
projector = build_vision_projector(config,Mcfg)

if Tcfg.checkpoint_status:
    checkpoint = torch.load(Tcfg.checkpoint_path)
    peft_LLMs = PeftModel.from_pretrained(LLMs_model, checkpoint["peft_LLMs_lora_path"])
    projector.load_state_dict(checkpoint['projector_state_dict'])
else:
    peft_LLMs = PeftModel.from_pretrained(LLMs_model, Mcfg.model_save_path)
    projector.load_state_dict(torch.load(Mcfg.projetc_save_path))
projector.to(peft_LLMs.device)

if Tcfg.fusion_type!=FusionType.LINEAR:
    crossmultiatt = DynamicRouter(view_types=Fcfg.view_types)
    crossmultiatt.load_state_dict(torch.load(Fcfg.multia_save_path, map_location=device1))
else:
    crossmultiatt = None

projector.eval()
peft_LLMs.eval()
if Tcfg.fusion_type!=FusionType.LINEAR: crossmultiatt.eval()
for i in range(1):
    save_path = Tcfg.Generate_files
    file_name = f"small_maira2_3w2_high_EPOCH3_avg_standard{i+1}.csv"
    os.makedirs(save_path, exist_ok=True)
    progress_bar_train = tqdm(dataloader_test, leave=True, mininterval=1.0)
    results = []
    for step, batch in enumerate(progress_bar_train):
        img_CF_feature,img_PF_feature,img_CL_feature,ind_att,ind,tech_att,tech,comp_att,comp,pf_att,pf = batch
        if Tcfg.fusion_type == FusionType.LINEAR:
            r_dict = fuse1_4096_test(Tcfg,peft_LLMs,projector,img_CF_feature,img_PF_feature,img_CL_feature,ind_att,ind,tech_att,tech,comp_att,comp,pf_att,pf)
        else:
            r_dict = fuse23_test(Tcfg,peft_LLMs,crossmultiatt,projector,img_CF_feature,img_PF_feature,img_CL_feature,ind_att,ind,tech_att,tech,comp_att,comp,pf_att,pf)
        att = r_dict['attention'].to(peft_LLMs.device)
        input_embeddings = r_dict['input_embeding'].to(peft_LLMs.device)
        print(f"最终的输入嵌入形状为{input_embeddings.shape},最终输入的注意力形状为{att.shape}")
        with torch.no_grad():
            outputs = peft_LLMs.generate(inputs_embeds=input_embeddings.to(torch.bfloat16),
                                        attention_mask = att,
                                        max_new_tokens = 256,
                                        min_new_tokens = 47, 
                                        #  do_sample=False,
                                        repetition_penalty=1.2,     # 加大重复惩罚
                                        do_sample=True,             # 启用采样
                                        temperature=0.9,            # 设置温度
                                        top_k=50,                   # top-k采样
                                        top_p=0.6,                  # top-p采样
                                        )
            temp = [LLMs_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            for i in range(len(temp)):
                results.append(temp[i])
            print(len(results))
    df = pd.DataFrame({"Report Impression": results})
    # df['study_id'] = dataset_test_new['study_id']
    df['study_id'] = data_test_avg['study_id']
    full_save_path = os.path.join(save_path, file_name)
    df.to_csv(full_save_path, index=False)

    print(f"数据已成功写入 {full_save_path} 文件。")



