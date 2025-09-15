# aligned_dataset.py
from multiprocessing import process
from networkx import predecessor
from torch.utils.data import Dataset
import torch
    

class AlignDataset_train_all(Dataset):
    def __init__(self, processor, model, data, imgs_CF, imgs_PF, imgs_CL):
        self.processor = processor
        self.model = model
        self.imgs_CF = imgs_CF
        self.imgs_PF = imgs_PF
        self.imgs_CL = imgs_CL
        self.data = data
        print(len(self.data))

    def __len__(self):
        return len(self.imgs_CF)

    def __getitem__(self, idx):
        #文本嵌入
        #target_embed
        tar_embeds = self.model.get_input_embeddings()(self.data['labels'][idx])
        #indication_embed
        ind_embeds = self.model.get_input_embeddings()(self.data['indication'][idx])
        #technique_embed
        tech_embeds = self.model.get_input_embeddings()(self.data['technology'][idx])
        #comparison_embeds
        comp_embeds = self.model.get_input_embeddings()(self.data['comparison'][idx])
        #prior_f
        prior_f_embeds = self.model.get_input_embeddings()(self.data['prior'][idx])
        # 图像嵌入
        #1
        if self.imgs_CF[idx] is None:
            img_CF_feature = "None"
        else:
            print(f"CF路径：{self.imgs_CF[idx]}")
            img_CF_feature = torch.Tensor(self.processor(self.imgs_CF[idx])).squeeze(0)[1:,:]
        #2
        if self.imgs_PF[idx] is None:
            img_PF_feature = "None"
        else:
            img_PF_feature = torch.Tensor(self.processor(self.imgs_PF[idx])).squeeze(0)[1:,:]   
        #3
        if self.imgs_CL[idx] is None:
            img_CL_feature = "None"
        else:
            img_CL_feature = torch.Tensor(self.processor(self.imgs_CL[idx])).squeeze(0)[1:,:]
        #扩充labels(好像不能再这里扩充labels，因为后面会对输入进行缩放，要等缩放之后在进行),直接在这里缩放试试,不行跑这个要两张显存80G的卡

        return img_CF_feature, img_PF_feature, img_CL_feature, self.data['labels'][idx], self.data['attention_mask'][idx], tar_embeds,  self.data['indication_att'][idx],ind_embeds, self.data['technology_att'][idx],tech_embeds, self.data['comparison_att'][idx],comp_embeds, self.data['prior_att'][idx],prior_f_embeds



class AlignDataset_batches_all(Dataset):
    def __init__(self, processor, model, data, imgs_CF, imgs_PF, imgs_CL):
        self.processor = processor
        self.model = model
        self.imgs_CF = imgs_CF
        self.imgs_PF = imgs_PF
        self.imgs_CL = imgs_CL
        self.data = data
        self.default_feature = torch.zeros(1369, 768)  # 共享的零张量
        print(len(self.data))

    def __len__(self):
        return len(self.imgs_CF)

    def __getitem__(self, idx):
        #文本嵌入
        #target_embed
        tar_embeds = self.model.get_input_embeddings()(self.data['labels'][idx])
        #indication_embed
        ind_embeds = self.model.get_input_embeddings()(self.data['indication'][idx])
        #technique_embed
        tech_embeds = self.model.get_input_embeddings()(self.data['technology'][idx])
        #comparison_embeds
        comp_embeds = self.model.get_input_embeddings()(self.data['comparison'][idx])
        #prior_f
        prior_f_embeds = self.model.get_input_embeddings()(self.data['prior'][idx])
        # 图像嵌入
        valid_mask = {
            'cf': int(self.imgs_CF[idx] is not None),
            'pf': int(self.imgs_PF[idx] is not None),
            'cl': int(self.imgs_CL[idx] is not None)
        }
        placeholder = torch.zeros_like(self.default_feature)


        #1
        if self.imgs_CF[idx] is None:
            img_CF_feature = placeholder
        else:
            img_CF_feature = torch.Tensor(self.processor(self.imgs_CF[idx])).squeeze(0)[1:,:]
        #2
        if self.imgs_PF[idx] is None:
            img_PF_feature = placeholder
        else:
            img_PF_feature = torch.Tensor(self.processor(self.imgs_PF[idx])).squeeze(0)[1:,:]   
        #3
        if self.imgs_CL[idx] is None:
            img_CL_feature = placeholder
        else:
            img_CL_feature = torch.Tensor(self.processor(self.imgs_CL[idx])).squeeze(0)[1:,:]
        #扩充labels(好像不能再这里扩充labels，因为后面会对输入进行缩放，要等缩放之后在进行),直接在这里缩放试试,不行跑这个要两张显存80G的卡

        return img_CF_feature, img_PF_feature, img_CL_feature, self.data['labels'][idx], self.data['attention_mask'][idx], tar_embeds,  self.data['indication_att'][idx],ind_embeds, self.data['technology_att'][idx],tech_embeds, self.data['comparison_att'][idx],comp_embeds, self.data['prior_att'][idx],prior_f_embeds,torch.tensor([valid_mask['cf'], valid_mask['pf'], valid_mask['cl']])

class AlignDataset_test_all(Dataset):
    def __init__(self, processor, model, data, imgs_CF, imgs_PF, imgs_CL):
        self.processor = processor
        self.model = model
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
        ind_embeds = self.model.get_input_embeddings()(self.data['indication'][idx])
        #technique_embed
        tech_embeds = self.model.get_input_embeddings()(self.data['technology'][idx])
        #comparison_embeds
        comp_embeds = self.model.get_input_embeddings()(self.data['comparison'][idx])
        #prior_f
        prior_f_embeds = self.model.get_input_embeddings()(self.data['prior'][idx])
        # 图像嵌入
        #1
        if self.imgs_CF[idx] is None:
            img_CF_feature = "None"
        else:
            img_CF_feature = torch.Tensor(self.processor(self.imgs_CF[idx])).squeeze(0)[1:,:]
        #2
        if self.imgs_PF[idx] is None:
            img_PF_feature = "None"
        else:
            img_PF_feature = torch.Tensor(self.processor(self.imgs_PF[idx])).squeeze(0)[1:,:]   
        #3
        if self.imgs_CL[idx] is None:
            img_CL_feature = "None"
        else:
            img_CL_feature = torch.Tensor(self.processor(self.imgs_CL[idx])).squeeze(0)[1:,:]
        #扩充labels(好像不能再这里扩充labels，因为后面会对输入进行缩放，要等缩放之后在进行),直接在这里缩放试试,不行跑这个要两张显存80G的卡

        return img_CF_feature, img_PF_feature, img_CL_feature, self.data['indication_att'][idx],ind_embeds, self.data['technology_att'][idx],tech_embeds, self.data['comparison_att'][idx],comp_embeds, self.data['prior_att'][idx],prior_f_embeds



#__________________________________________________________________________________________________________________________________________________________________

class TestDataset_image_indication(Dataset):
    def __init__(self, processor,model,data, image_paths):
        self.img_paths = image_paths
        self.data = data
        self.processor = processor
        self.model = model
        print(len(self.img_paths))
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 图像嵌入
        image_feature = torch.Tensor(self.processor(self.img_paths[idx])).squeeze(0)[:1369,:]
        #indication_embed
        ind_embeds = self.model.get_input_embeddings()(self.data['indication'][idx])

        return image_feature , self.data['ind_att'],ind_embeds



class AlignDataset_test_image_only(Dataset):
    def __init__(self, processor, image_paths):
        self.img_paths = image_paths
        self.processor = processor
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 图像嵌入
        image_feature = torch.Tensor(self.processor(self.img_paths[idx])).squeeze(0)[1:,:]

        return image_feature









# 主程序中使用
# from trained_dataset import AlignDataset

# dataset = AlignDataset(
#     processor=processor,
#     model = model,
#     data = data, 
#     imgs_CF, 
#     imgs_PF, 
#     imgs_CL, 
#     labels, 
#     att, 
#     ind, 
#     tech, 
#     comp, 
#     prior_f
# )