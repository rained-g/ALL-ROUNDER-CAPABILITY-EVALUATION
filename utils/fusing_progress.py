import torch
from utils.configset import TrainingConfig,FusionType
import torch.nn.functional as F

def fuse1(TrainingConfig,model,LLMs_token,projector,img_CF_feature,img_PF_feature,img_CL_feature,labels,att,embeds,ind_att,ind,tech_att,tech,comp_att,comp,pf_att,pf):
    result_dict = {}
    # 获取视觉嵌入-1
    img_CF_feature = img_CF_feature.to(next(projector.parameters()).device)
    # print(img_CF_feature.shape)
    vision_CF_embeds = projector(img_CF_feature)
    # print(vision_CF_embeds.shape)
    del img_CF_feature

    #扩展提示嵌入以匹配视觉特征
    expand_sys_embeds = TrainingConfig.prompt_embed['sys_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_user_embeds = TrainingConfig.prompt_embed['user_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_usual1_embeds = TrainingConfig.prompt_embed['usual1_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_pf_embeds = TrainingConfig.prompt_embed['P_Re_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Tech_embeds = TrainingConfig.prompt_embed['Tech_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Comp_embeds = TrainingConfig.prompt_embed['Comp_embeds'].expand(TrainingConfig.batch_size,-1,-1)

    # 确定输入类型
    if isinstance(img_PF_feature, torch.Tensor) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL + PF'
    elif isinstance(img_PF_feature, torch.Tensor) and (not isinstance(img_CL_feature, torch.Tensor)):
        input_type = 'CF + PF'
    elif (not isinstance(img_PF_feature, torch.Tensor)) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL'
    else:
        input_type = 'CF only'

    result_dict['input_type'] = input_type
    #结合系统提示以及CF,CL.PF
    #CF
    input_embeddings = torch.cat([expand_sys_embeds,vision_CF_embeds.to(model.device)],dim=1)
    del expand_sys_embeds
    input_embeddings = torch.cat([input_embeddings,expand_usual1_embeds],dim=1)
    #CL
    if img_CL_feature != ('None',):
        expand_CL_embeds = TrainingConfig.prompt_embed['C_lateral_embed'].expand(vision_CF_embeds.size(0),-1,-1)
        img_CL_feature = img_CL_feature.to(next(projector.parameters()).device)
        vision_CL_embeds = projector(img_CL_feature)
        del img_CL_feature
        input_embeddings = torch.cat([input_embeddings,expand_CL_embeds],dim=1)
        input_embeddings = torch.cat([input_embeddings,vision_CL_embeds.to(model.device)],dim=1)
        del expand_CL_embeds
        # del vision_CL_embeds
        input_embeddings = torch.cat([input_embeddings,expand_usual1_embeds],dim=1)
    #PF
    if img_PF_feature != ('None',):
        expand_PF_embeds = TrainingConfig.prompt_embed['P_frontal_embeds'].expand(vision_CF_embeds.size(0),-1,-1)
        img_PF_feature = img_PF_feature.to(next(projector.parameters()).device)
        vision_PF_embeds = projector(img_PF_feature)
        del img_PF_feature
        input_embeddings = torch.cat([input_embeddings,expand_PF_embeds],dim=1)
        input_embeddings = torch.cat([input_embeddings,vision_PF_embeds.to(model.device)],dim=1)
        del expand_PF_embeds
        # del vision_PF_embeds
        input_embeddings = torch.cat([input_embeddings,expand_usual1_embeds],dim=1)
        del expand_usual1_embeds
        # del vision_CF_embeds

    # print(f"缩放前的输入维度:{input_embeddings.shape}")
    # 定义目标长度
    target_length = 1712
    
    # target_length = input_embeddings.shape[1] / 2.4
    if target_length < input_embeddings.shape[1]:

        # 使用插值进行线性缩放
        input_embeddings = input_embeddings.permute(0, 2, 1)  # 形状: (1, 4096, len(input_embedings))
        # 使用线性插值缩放
        scaled_input_embeddings = F.interpolate(input_embeddings, size=target_length, mode='linear', align_corners=False)  # 形状: (1,4096,1530)
        del input_embeddings
        # 恢复原始形状
        input_embeddings = scaled_input_embeddings.permute(0, 2, 1)  # 形状: (1,1530, 4096)
        del scaled_input_embeddings
    # print(f"缩放后的输入维度:{input_embeddings.shape}")
    att_mask = F.pad(TrainingConfig.attention_mask['P_Re_att'], (input_embeddings.shape[1],0), value=1)
    #结合pf
    input_embeddings = torch.cat([input_embeddings,expand_pf_embeds],dim=1)
    input_embeddings = torch.cat([input_embeddings,pf.to(model.device)],dim=1)
    del expand_pf_embeds
    del pf
    #结合用户提示以及indication
    input_embeddings = torch.cat([input_embeddings,expand_user_embeds],dim=1)
    input_embeddings = torch.cat([input_embeddings,ind.to(model.device)],dim=1)
    del expand_user_embeds
    del ind
    #结合technique
    input_embeddings = torch.cat([input_embeddings,expand_Tech_embeds],dim=1)
    input_embeddings = torch.cat([input_embeddings,tech.to(model.device)],dim=1)
    del expand_Tech_embeds
    del tech
    #结合comparison
    input_embeddings = torch.cat([input_embeddings,expand_Comp_embeds],dim=1)
    input_embeddings = torch.cat([input_embeddings,comp.to(model.device)],dim=1)
    del expand_Comp_embeds
    del comp

    components_att = [
        att_mask,
        pf_att,
        TrainingConfig.attention_mask['user_att'],
        ind_att,
        TrainingConfig.attention_mask['Tech_att'],
        tech_att,
        TrainingConfig.attention_mask['Comp_att'],
        comp_att,
        att
    ]

    #将labels以及attention进行改变
    att_mask = torch.cat(components_att,dim=-1)
    labels = F.pad(labels, (input_embeddings.shape[1],0), value=-100)
    labels = labels.masked_fill(labels == LLMs_token.pad_token_id, -100)
    labels = labels.to(model.device, dtype=torch.int64)
    #将目标文本进行嵌入
    input_embeddings = torch.cat((input_embeddings,embeds.to(model.device)),dim=1)
    del embeds
    print(f"input_embeddings的维度为{input_embeddings.shape},labels的维度为{labels.shape},att的维度为{att_mask.shape}")
    # print(labels[:,-140:])
    # 手动释放缓存
    torch.cuda.empty_cache() 

    result_dict['input_embeding'] = input_embeddings
    result_dict['attention'] = att_mask
    result_dict['labels'] = labels
    return result_dict

#____________________________________________________________________________________________________________________________________________________________________fuse1

def fuse1_4096(TrainingConfig,model,LLMs_token,projector,img_CF_feature,img_PF_feature,img_CL_feature,labels,att,embeds,ind_att,ind,tech_att,tech,comp_att,comp,pf_att,pf):
    result_dict = {}
    # 获取视觉嵌入-1
    img_CF_feature = img_CF_feature.to(next(projector.parameters()).device)
    # print(img_CF_feature.shape)
    vision_CF_embeds = projector(img_CF_feature)
    # print(vision_CF_embeds.shape)
    del img_CF_feature

    #扩展提示嵌入以匹配视觉特征
    expand_sys_embeds = TrainingConfig.prompt_embed['sys_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_user_embeds = TrainingConfig.prompt_embed['user_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_usual1_embeds = TrainingConfig.prompt_embed['usual1_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_pf_embeds = TrainingConfig.prompt_embed['P_Re_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Tech_embeds = TrainingConfig.prompt_embed['Tech_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Comp_embeds = TrainingConfig.prompt_embed['Comp_embeds'].expand(TrainingConfig.batch_size,-1,-1)

    # 确定输入类型
    if isinstance(img_PF_feature, torch.Tensor) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL + PF'
    elif isinstance(img_PF_feature, torch.Tensor) and (not isinstance(img_CL_feature, torch.Tensor)):
        input_type = 'CF + PF'
    elif (not isinstance(img_PF_feature, torch.Tensor)) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL'
    else:
        input_type = 'CF only'

    result_dict['input_type'] = input_type

    #结合系统提示以及CF,CL.PF
    # 定义目标长度
    # max_length = 3752
    max_length = 3755

    if input_type == 'CF + CL + PF':
        expand_CL_embeds = TrainingConfig.prompt_embed['C_lateral_embed'].expand(vision_CF_embeds.size(0),-1,-1)
        img_CL_feature = img_CL_feature.to(next(projector.parameters()).device)
        vision_CL_embeds = projector(img_CL_feature)
        del img_CL_feature

        expand_PF_embeds = TrainingConfig.prompt_embed['P_frontal_embeds'].expand(vision_CF_embeds.size(0),-1,-1)
        img_PF_feature = img_PF_feature.to(next(projector.parameters()).device)
        vision_PF_embeds = projector(img_PF_feature)
        del img_PF_feature

        target_length = (max_length - expand_sys_embeds.shape[1] - 3*expand_usual1_embeds.shape[1] - expand_CL_embeds.shape[1] - expand_PF_embeds.shape[1]) // 3

        # 使用插值进行线性缩放
        vision_CF_embeds = vision_CF_embeds.permute(0, 2, 1)  # 形状: (1, 4096, len(input_embedings))
        vision_CL_embeds = vision_CL_embeds.permute(0, 2, 1)
        vision_PF_embeds = vision_PF_embeds.permute(0, 2, 1)
        # 使用线性插值缩放
        scaled_vision_CF_embeds = F.interpolate(vision_CF_embeds, size=target_length, mode='linear', align_corners=False)  # 形状: (1,4096,1530)
        del vision_CF_embeds
        scaled_vision_CL_embeds = F.interpolate(vision_CL_embeds, size=target_length, mode='linear', align_corners=False)  # 形状: (1,4096,1530)
        del vision_CL_embeds
        scaled_vision_PF_embeds = F.interpolate(vision_PF_embeds, size=target_length, mode='linear', align_corners=False)  # 形状: (1,4096,1530)
        del vision_PF_embeds
        # 恢复原始形状
        vision_CF_embeds = scaled_vision_CF_embeds.permute(0, 2, 1)  # 形状: (1,1530, 4096)
        del scaled_vision_CF_embeds
        vision_CL_embeds = scaled_vision_CL_embeds.permute(0, 2, 1)  # 形状: (1,1530, 4096)
        del scaled_vision_CL_embeds 
        vision_PF_embeds = scaled_vision_PF_embeds.permute(0, 2, 1)  # 形状: (1,1530, 4096)
        del scaled_vision_PF_embeds 
        
        input_embeddings = torch.cat([expand_sys_embeds,vision_CF_embeds.to(model.device)],dim=1)
        del expand_sys_embeds
        del vision_CF_embeds
        input_embeddings = torch.cat([input_embeddings,expand_usual1_embeds],dim=1)

        input_embeddings = torch.cat([input_embeddings,expand_CL_embeds],dim=1)
        input_embeddings = torch.cat([input_embeddings,vision_CL_embeds.to(model.device)],dim=1)
        del expand_CL_embeds
        del vision_CL_embeds
        input_embeddings = torch.cat([input_embeddings,expand_usual1_embeds],dim=1)

        input_embeddings = torch.cat([input_embeddings,expand_PF_embeds],dim=1)
        input_embeddings = torch.cat([input_embeddings,vision_PF_embeds.to(model.device)],dim=1)
        del expand_PF_embeds
        del vision_PF_embeds
        input_embeddings = torch.cat([input_embeddings,expand_usual1_embeds],dim=1)
        del expand_usual1_embeds

    else:
        #CF
        input_embeddings = torch.cat([expand_sys_embeds,vision_CF_embeds.to(model.device)],dim=1)
        del expand_sys_embeds
        input_embeddings = torch.cat([input_embeddings,expand_usual1_embeds],dim=1)
        #CL
        if img_CL_feature != ('None',):
            expand_CL_embeds = TrainingConfig.prompt_embed['C_lateral_embed'].expand(vision_CF_embeds.size(0),-1,-1)
            img_CL_feature = img_CL_feature.to(next(projector.parameters()).device)
            vision_CL_embeds = projector(img_CL_feature)
            del img_CL_feature
            input_embeddings = torch.cat([input_embeddings,expand_CL_embeds],dim=1)
            input_embeddings = torch.cat([input_embeddings,vision_CL_embeds.to(model.device)],dim=1)
            del expand_CL_embeds
            del vision_CL_embeds
            input_embeddings = torch.cat([input_embeddings,expand_usual1_embeds],dim=1)
        #PF
        if img_PF_feature != ('None',):
            expand_PF_embeds = TrainingConfig.prompt_embed['P_frontal_embeds'].expand(vision_CF_embeds.size(0),-1,-1)
            img_PF_feature = img_PF_feature.to(next(projector.parameters()).device)
            vision_PF_embeds = projector(img_PF_feature)
            del img_PF_feature
            input_embeddings = torch.cat([input_embeddings,expand_PF_embeds],dim=1)
            input_embeddings = torch.cat([input_embeddings,vision_PF_embeds.to(model.device)],dim=1)
            del expand_PF_embeds
            del vision_PF_embeds
            input_embeddings = torch.cat([input_embeddings,expand_usual1_embeds],dim=1)
            del expand_usual1_embeds
            del vision_CF_embeds

    att_mask = F.pad(TrainingConfig.attention_mask['P_Re_att'], (input_embeddings.shape[1],0), value=1)
    #结合pf
    input_embeddings = torch.cat([input_embeddings,expand_pf_embeds],dim=1)
    input_embeddings = torch.cat([input_embeddings,pf.to(model.device)],dim=1)
    del expand_pf_embeds
    del pf
    #结合用户提示以及indication
    input_embeddings = torch.cat([input_embeddings,expand_user_embeds],dim=1)
    input_embeddings = torch.cat([input_embeddings,ind.to(model.device)],dim=1)
    del expand_user_embeds
    del ind
    #结合technique
    input_embeddings = torch.cat([input_embeddings,expand_Tech_embeds],dim=1)
    input_embeddings = torch.cat([input_embeddings,tech.to(model.device)],dim=1)
    del expand_Tech_embeds
    del tech
    #结合comparison
    input_embeddings = torch.cat([input_embeddings,expand_Comp_embeds],dim=1)
    input_embeddings = torch.cat([input_embeddings,comp.to(model.device)],dim=1)
    del expand_Comp_embeds
    del comp

    components_att = [
        att_mask,
        pf_att,
        TrainingConfig.attention_mask['user_att'],
        ind_att,
        TrainingConfig.attention_mask['Tech_att'],
        tech_att,
        TrainingConfig.attention_mask['Comp_att'],
        comp_att,
        att
    ]

    #将labels以及attention进行改变
    att_mask = torch.cat(components_att,dim=-1)
    labels = F.pad(labels, (input_embeddings.shape[1],0), value=-100)
    labels = labels.masked_fill(labels == LLMs_token.pad_token_id, -100)
    labels = labels.to(model.device, dtype=torch.int64)
    #将目标文本进行嵌入
    input_embeddings = torch.cat((input_embeddings,embeds.to(model.device)),dim=1)
    del embeds
    print(f"input_embeddings的维度为{input_embeddings.shape},labels的维度为{labels.shape},att的维度为{att_mask.shape}")
    # print(labels[:,-140:])
    # 手动释放缓存
    torch.cuda.empty_cache() 

    result_dict['input_embeding'] = input_embeddings
    result_dict['attention'] = att_mask
    result_dict['labels'] = labels
    return result_dict

#____________________________________________________________________________________________________________________________________________________________________fuse1_4096

def fuse23(TrainingConfig,model,LLMs_token,fusion,projector,img_CF_feature,img_PF_feature,img_CL_feature,labels,att,embeds,ind_att,ind,tech_att,tech,comp_att,comp,pf_att,pf):
    result_dict = {}
    views = {}
    print(f"先前文本形状：{pf.shape},注意力格式：{pf_att}")
    #扩展提示嵌入以匹配视觉特征
    expand_sys_embeds = TrainingConfig.prompt_embed['sys_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_user_embeds = TrainingConfig.prompt_embed['user_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_usual1_embeds = TrainingConfig.prompt_embed['usual1_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_pf_embeds = TrainingConfig.prompt_embed['P_Re_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Tech_embeds = TrainingConfig.prompt_embed['Tech_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Comp_embeds = TrainingConfig.prompt_embed['Comp_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    
    # 确定输入类型
    if isinstance(img_PF_feature, torch.Tensor) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL + PF'
        views['cf'] = img_CF_feature.to(next(fusion.parameters()).device)    #[1,1369,768]
        views['cl'] = img_CL_feature.to(next(fusion.parameters()).device)    #[1,1369,768]
        views['pf'] = img_PF_feature.to(next(fusion.parameters()).device)    #[1,1369,768]
    elif isinstance(img_PF_feature, torch.Tensor) and (not isinstance(img_CL_feature, torch.Tensor)):
        input_type = 'CF + PF'
        views['cf'] = img_CF_feature.to(next(fusion.parameters()).device)    #[1,1369,768]
        views['pf'] = img_PF_feature.to(next(fusion.parameters()).device)    #[1,1369,768]
    elif (not isinstance(img_PF_feature, torch.Tensor)) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL'
        views['cf'] = img_CF_feature.to(next(fusion.parameters()).device)    #[1,1369,768]
        views['cl'] = img_CL_feature.to(next(fusion.parameters()).device)     #[1,1369,768]
    else:
        input_type = 'CF only'
        views['cf'] = img_CF_feature.to(next(fusion.parameters()).device)    #[1,1369,768]
    result_dict['input_type'] = input_type

    #判断注意力方法
    if TrainingConfig.fusion_type == FusionType.SELF_ATTN:
        #拼接图像
        temp_views = []
        for key in views.keys():
            temp_views.append(views[key])
        image_features = torch.cat(temp_views,dim=1) if len(temp_views) > 1 else temp_views[0]
        #注意力机制压缩融合输入图像信息
        image_features = fusion(image_features)
    elif TrainingConfig.fusion_type == FusionType.CROSS_ATTN:
        image_features = fusion(features_dict=views)
    
    #投影到文本空间
    vision_embeds = projector(image_features.to(next(projector.parameters()).device))
    vision_embeds = vision_embeds.to(model.device)
    if input_type == 'CF only':
        input_embeddings = torch.cat([expand_sys_embeds,vision_embeds,expand_usual1_embeds],dim=1)
        att_mask = F.pad(TrainingConfig.attention_mask['sys_att'], (0,vision_embeds.shape[1]), value=1)
        att_mask = torch.cat([att_mask, TrainingConfig.attention_mask['usual1_att']], dim=-1)
        expand_usual1_embeds = expand_usual1_embeds.detach()

    elif input_type == 'CF + CL + PF':
        expand_sys1_embeds = TrainingConfig.prompt_embed['sys1_embeds'].expand(TrainingConfig.batch_size,-1,-1)
        input_embeddings = torch.cat([expand_sys1_embeds,vision_embeds,expand_usual1_embeds],dim=1)
        att_mask = F.pad(TrainingConfig.attention_mask['sys1_att'], (0,vision_embeds.shape[1]), value=1)
        att_mask = torch.cat([att_mask, TrainingConfig.attention_mask['usual1_att']], dim=-1)
        expand_sys1_embeds = expand_sys1_embeds.detach()

    elif input_type == 'CF + PF':
        expand_sys2_embeds = TrainingConfig.prompt_embed['sys2_embeds'].expand(TrainingConfig.batch_size,-1,-1)
        input_embeddings = torch.cat([expand_sys2_embeds,vision_embeds,expand_usual1_embeds],dim=1)
        att_mask = F.pad(TrainingConfig.attention_mask['sys2_att'], (0,vision_embeds.shape[1]), value=1)
        att_mask = torch.cat([att_mask, TrainingConfig.attention_mask['usual1_att']], dim=-1)
        expand_sys2_embeds = expand_sys2_embeds.detach()

    else:
        expand_sys3_embeds = TrainingConfig.prompt_embed['sys3_embeds'].expand(TrainingConfig.batch_size,-1,-1)
        input_embeddings = torch.cat([expand_sys3_embeds,vision_embeds,expand_usual1_embeds],dim=1)
        att_mask = F.pad(TrainingConfig.attention_mask['sys3_att'], (0,vision_embeds.shape[1]), value=1)
        att_mask = torch.cat([att_mask, TrainingConfig.attention_mask['usual1_att']], dim=-1)
        expand_sys3_embeds = expand_sys3_embeds.detach()
        
    # print(input_type)
    pf = pf.to(model.device).detach()
    ind = ind.to(model.device).detach()
    tech = tech.to(model.device).detach()
    comp = comp.to(model.device).detach()
    components = [
        input_embeddings,
        expand_pf_embeds,
        pf,
        expand_user_embeds,
        ind,
        expand_Tech_embeds,
        tech,
        expand_Comp_embeds,
        comp,
    ]
    components_att = [
        att_mask,
        TrainingConfig.attention_mask['P_Re_att'],
        pf_att,
        TrainingConfig.attention_mask['user_att'],
        ind_att,
        TrainingConfig.attention_mask['Tech_att'],
        tech_att,
        TrainingConfig.attention_mask['Comp_att'],
        comp_att,
        att
    ]
    input_embeddings = torch.cat(components, dim=1)
    att_mask = torch.cat(components_att,dim=-1)

    #将labels以及attention进行改变
    labels = labels.masked_fill(labels == LLMs_token.pad_token_id, -100)
    labels = F.pad(labels, (input_embeddings.shape[1],0), value=-100)
    labels = labels.to(model.device, dtype=torch.int64)
    #将目标文本进行嵌入
    input_embeddings = torch.cat((input_embeddings,embeds.to(model.device)),dim=1)
    embeds = embeds.detach()

    # 手动释放缓存 
    torch.cuda.empty_cache()


    result_dict['input_embeding'] = input_embeddings
    result_dict['attention'] = att_mask
    result_dict['labels'] = labels
    return result_dict

#____________________________________________________________________________________________________________________________________________________________________fuse2 and 3

def fuse4(TrainingConfig,model,LLMs_token,fusion,projector,img_CF_feature,img_PF_feature,img_CL_feature,labels,att,embeds,ind_att,ind,tech_att,tech,comp_att,comp,pf_att,pf):
    result_dict = {}
    views = {}

    #扩展提示嵌入以匹配视觉特征
    expand_sys_embeds = TrainingConfig.prompt_embed['sys_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_user_embeds = TrainingConfig.prompt_embed['user_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_usual1_embeds = TrainingConfig.prompt_embed['usual1_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_pf_embeds = TrainingConfig.prompt_embed['P_Re_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Tech_embeds = TrainingConfig.prompt_embed['Tech_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Comp_embeds = TrainingConfig.prompt_embed['Comp_embeds'].expand(TrainingConfig.batch_size,-1,-1)

    # 确定输入类型
    if isinstance(img_PF_feature, torch.Tensor) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL + PF'
        views['cf'] = img_CF_feature.to(next(fusion.parameters()).device)    #[1,1369,768]
        views['cl'] = img_CL_feature.to(next(fusion.parameters()).device)    #[1,1369,768]
        views['pf'] = img_PF_feature.to(next(fusion.parameters()).device)    #[1,1369,768]
    elif isinstance(img_PF_feature, torch.Tensor) and (not isinstance(img_CL_feature, torch.Tensor)):
        input_type = 'CF + PF'
        views['cf'] = img_CF_feature.to(next(fusion.parameters()).device)    #[1,1369,768]
        views['pf'] = img_PF_feature.to(next(fusion.parameters()).device)    #[1,1369,768]
    elif (not isinstance(img_PF_feature, torch.Tensor)) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL'
        views['cf'] = img_CF_feature.to(next(fusion.parameters()).device)    #[1,1369,768]
        views['cl'] = img_CL_feature.to(next(fusion.parameters()).device)     #[1,1369,768]
    else:
        input_type = 'CF only'
        views['cf'] = img_CF_feature.to(next(fusion.parameters()).device)    #[1,1369,768]
    result_dict['input_type'] = input_type

    #注意力机制融合信息
    fusion_result = fusion(features_dict=views)
    
    #投影到文本空间
    vision_embeds = projector(fusion_result.to(next(projector.parameters()).device))
    vision_embeds = vision_embeds.to(model.device)
    if input_type == 'CF only':
        input_embeddings = torch.cat([expand_sys_embeds,vision_embeds,expand_usual1_embeds],dim=1)
        att_mask = F.pad(TrainingConfig.attention_mask['sys_att'], (0,vision_embeds.shape[1]), value=1)
        att_mask = torch.cat([att_mask, TrainingConfig.attention_mask['usual1_att']], dim=-1)
        expand_usual1_embeds = expand_usual1_embeds.detach()

    elif input_type == 'CF + CL + PF':
        expand_sys1_embeds = TrainingConfig.prompt_embed['sys1_embeds'].expand(TrainingConfig.batch_size,-1,-1)
        input_embeddings = torch.cat([expand_sys1_embeds,vision_embeds,expand_usual1_embeds],dim=1)
        att_mask = F.pad(TrainingConfig.attention_mask['sys1_att'], (0,vision_embeds.shape[1]), value=1)
        att_mask = torch.cat([att_mask, TrainingConfig.attention_mask['usual1_att']], dim=-1)
        expand_sys1_embeds = expand_sys1_embeds.detach()

    elif input_type == 'CF + PF':
        expand_sys2_embeds = TrainingConfig.prompt_embed['sys2_embeds'].expand(TrainingConfig.batch_size,-1,-1)
        input_embeddings = torch.cat([expand_sys2_embeds,vision_embeds,expand_usual1_embeds],dim=1)
        att_mask = F.pad(TrainingConfig.attention_mask['sys2_att'], (0,vision_embeds.shape[1]), value=1)
        att_mask = torch.cat([att_mask, TrainingConfig.attention_mask['usual1_att']], dim=-1)
        expand_sys2_embeds = expand_sys2_embeds.detach()

    else:
        expand_sys3_embeds = TrainingConfig.prompt_embed['sys3_embeds'].expand(TrainingConfig.batch_size,-1,-1)
        input_embeddings = torch.cat([expand_sys3_embeds,vision_embeds,expand_usual1_embeds],dim=1)
        att_mask = F.pad(TrainingConfig.attention_mask['sys3_att'], (0,vision_embeds.shape[1]), value=1)
        att_mask = torch.cat([att_mask, TrainingConfig.attention_mask['usual1_att']], dim=-1)
        expand_sys3_embeds = expand_sys3_embeds.detach()
        
    # print(input_type)
    pf = pf.to(model.device).detach()
    ind = ind.to(model.device).detach()
    tech = tech.to(model.device).detach()
    comp = comp.to(model.device).detach()

    components = [
        input_embeddings,
        expand_pf_embeds,
        pf,
        expand_user_embeds,
        ind,
        expand_Tech_embeds,
        tech,
        expand_Comp_embeds,
        comp,
    ]
    # print(P_Re_att.shape,pf_att.shape,user_att.shape,ind_att.shape,Tech_att.shape,tech_att.shape,Comp_att.shape,comp_att.shape, att.shape)
    components_att = [
        att_mask,
        TrainingConfig.attention_mask['P_Re_att'],
        pf_att,
        TrainingConfig.attention_mask['user_att'],
        ind_att,
        TrainingConfig.attention_mask['Tech_att'],
        tech_att,
        TrainingConfig.attention_mask['Comp_att'],
        comp_att,
        att
    ]
    # print(f"注意力掩码形状为{att_mask.shape}")
    input_embeddings = torch.cat(components, dim=1)
    att_mask = torch.cat(components_att,dim=-1)
    # print(f"注意力掩码形状为{att_mask[:,1416:1500]}")

    #将labels进行改变
    # print(LLMs_token.pad_token_id)
    labels = labels.masked_fill(labels == LLMs_token.pad_token_id, -100)
    labels = F.pad(labels, (input_embeddings.shape[1],0), value=-100)
    labels = labels.to(model.device, dtype=torch.int64)
    #将目标文本进行嵌入
    input_embeddings = torch.cat((input_embeddings,embeds.to(model.device)),dim=1)
    embeds = embeds.detach()

    # 手动释放缓存 
    torch.cuda.empty_cache()


    result_dict['input_embeding'] = input_embeddings
    result_dict['attention'] = att_mask
    result_dict['labels'] = labels
    return result_dict

#____________________________________________________________________________________________________________________________________________________________________fuse4


'''__________________________________________________________________________________________________________________________________________________________________________________________________________________________
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________
test'''

def fuse1_test(TrainingConfig,model,projector,img_CF_feature,img_PF_feature,img_CL_feature,ind_att,ind,tech_att,tech,comp_att,comp,pf_att,pf):
    result_dict = {}
    # 获取视觉嵌入-1
    img_CF_feature = img_CF_feature.to(next(projector.parameters()).device)
    # print(img_CF_feature.shape)
    vision_CF_embeds = projector(img_CF_feature)
    # print(vision_CF_embeds.shape)
    del img_CF_feature

    #扩展提示嵌入以匹配视觉特征
    expand_sys_embeds = TrainingConfig.prompt_embed['sys_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_user_embeds = TrainingConfig.prompt_embed['user_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_usual1_embeds = TrainingConfig.prompt_embed['usual1_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_pf_embeds = TrainingConfig.prompt_embed['P_Re_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Tech_embeds = TrainingConfig.prompt_embed['Tech_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Comp_embeds = TrainingConfig.prompt_embed['Comp_embeds'].expand(TrainingConfig.batch_size,-1,-1)

    # 确定输入类型
    if isinstance(img_PF_feature, torch.Tensor) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL + PF'
    elif isinstance(img_PF_feature, torch.Tensor) and (not isinstance(img_CL_feature, torch.Tensor)):
        input_type = 'CF + PF'
    elif (not isinstance(img_PF_feature, torch.Tensor)) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL'
    else:
        input_type = 'CF only'

    result_dict['input_type'] = input_type
    #结合系统提示以及CF,CL.PF
    #CF
    input_embeddings = torch.cat([expand_sys_embeds,vision_CF_embeds.to(model.device)],dim=1)
    del expand_sys_embeds
    input_embeddings = torch.cat([input_embeddings,expand_usual1_embeds],dim=1)
    #CL
    if img_CL_feature != ('None',):
        expand_CL_embeds = TrainingConfig.prompt_embed['C_lateral_embed'].expand(vision_CF_embeds.size(0),-1,-1)
        img_CL_feature = img_CL_feature.to(next(projector.parameters()).device)
        vision_CL_embeds = projector(img_CL_feature)
        del img_CL_feature
        input_embeddings = torch.cat([input_embeddings,expand_CL_embeds],dim=1)
        input_embeddings = torch.cat([input_embeddings,vision_CL_embeds.to(model.device)],dim=1)
        del expand_CL_embeds
        del vision_CL_embeds
        input_embeddings = torch.cat([input_embeddings,expand_usual1_embeds],dim=1)
    #PF
    if img_PF_feature != ('None',):
        expand_PF_embeds = TrainingConfig.prompt_embed['P_frontal_embeds'].expand(vision_CF_embeds.size(0),-1,-1)
        img_PF_feature = img_PF_feature.to(next(projector.parameters()).device)
        vision_PF_embeds = projector(img_PF_feature)
        del img_PF_feature
        input_embeddings = torch.cat([input_embeddings,expand_PF_embeds],dim=1)
        input_embeddings = torch.cat([input_embeddings,vision_PF_embeds.to(model.device)],dim=1)
        del expand_PF_embeds
        del vision_PF_embeds
        input_embeddings = torch.cat([input_embeddings,expand_usual1_embeds],dim=1)
        del expand_usual1_embeds
        del vision_CF_embeds

    # print(f"缩放前的输入维度:{input_embeddings.shape}")
    # 定义目标长度
    target_length = 1712
    # target_length = input_embeddings.shape[1] / 2.4
    if target_length < input_embeddings.shape[1]:
        # 使用插值进行线性缩放
        input_embeddings = input_embeddings.permute(0, 2, 1)  # 形状: (1, 4096, len(input_embedings))
        # 使用线性插值缩放
        scaled_input_embeddings = F.interpolate(input_embeddings, size=target_length, mode='linear', align_corners=False)  # 形状: (1,4096,1530)
        del input_embeddings
        # 恢复原始形状
        input_embeddings = scaled_input_embeddings.permute(0, 2, 1)  # 形状: (1,1530, 4096)
        del scaled_input_embeddings
    # print(f"缩放后的输入维度:{input_embeddings.shape}")
    att_mask = F.pad(TrainingConfig.attention_mask['P_Re_att'], (input_embeddings.shape[1],0), value=1)
    #结合pf
    input_embeddings = torch.cat([input_embeddings,expand_pf_embeds],dim=1)
    input_embeddings = torch.cat([input_embeddings,pf.to(model.device)],dim=1)
    del expand_pf_embeds
    del pf
    #结合用户提示以及indication
    input_embeddings = torch.cat([input_embeddings,expand_user_embeds],dim=1)
    input_embeddings = torch.cat([input_embeddings,ind.to(model.device)],dim=1)
    del expand_user_embeds
    del ind
    #结合technique
    input_embeddings = torch.cat([input_embeddings,expand_Tech_embeds],dim=1)
    input_embeddings = torch.cat([input_embeddings,tech.to(model.device)],dim=1)
    del expand_Tech_embeds
    del tech
    #结合comparison
    input_embeddings = torch.cat([input_embeddings,expand_Comp_embeds],dim=1)
    input_embeddings = torch.cat([input_embeddings,comp.to(model.device)],dim=1)
    del expand_Comp_embeds
    del comp

    components_att = [
        att_mask,
        pf_att,
        TrainingConfig.attention_mask['user_att'],
        ind_att,
        TrainingConfig.attention_mask['Tech_att'],
        tech_att,
        TrainingConfig.attention_mask['Comp_att'],
        comp_att,
    ]

    #将labels以及attention进行改变
    att_mask = torch.cat(components_att,dim=-1)
    # print(labels[:,-140:])
    # 手动释放缓存
    torch.cuda.empty_cache() 

    result_dict['input_embeding'] = input_embeddings
    result_dict['attention'] = att_mask
    return result_dict

#____________________________________________________________________________________________________________________________________________________________________fuse1


def fuse1_4096_test(TrainingConfig,model,projector,img_CF_feature,img_PF_feature,img_CL_feature,ind_att,ind,tech_att,tech,comp_att,comp,pf_att,pf):
    result_dict = {}
    # 获取视觉嵌入-1
    img_CF_feature = img_CF_feature.to(next(projector.parameters()).device)
    vision_CF_embeds = projector(img_CF_feature)
    del img_CF_feature

    #扩展提示嵌入以匹配视觉特征
    expand_sys_embeds = TrainingConfig.prompt_embed['sys_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_user_embeds = TrainingConfig.prompt_embed['user_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_usual1_embeds = TrainingConfig.prompt_embed['usual1_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_pf_embeds = TrainingConfig.prompt_embed['P_Re_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Tech_embeds = TrainingConfig.prompt_embed['Tech_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Comp_embeds = TrainingConfig.prompt_embed['Comp_embeds'].expand(TrainingConfig.batch_size,-1,-1)

    # 确定输入类型
    if isinstance(img_PF_feature, torch.Tensor) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL + PF'
    elif isinstance(img_PF_feature, torch.Tensor) and (not isinstance(img_CL_feature, torch.Tensor)):
        input_type = 'CF + PF'
    elif (not isinstance(img_PF_feature, torch.Tensor)) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL'
    else:
        input_type = 'CF only'

    result_dict['input_type'] = input_type

    #结合系统提示以及CF,CL.PF
    # 定义目标长度
    max_length = 3757

    if input_type == 'CF + CL + PF':
        expand_CL_embeds = TrainingConfig.prompt_embed['C_lateral_embed'].expand(vision_CF_embeds.size(0),-1,-1)
        img_CL_feature = img_CL_feature.to(next(projector.parameters()).device)
        vision_CL_embeds = projector(img_CL_feature)
        del img_CL_feature

        expand_PF_embeds = TrainingConfig.prompt_embed['P_frontal_embeds'].expand(vision_CF_embeds.size(0),-1,-1)
        img_PF_feature = img_PF_feature.to(next(projector.parameters()).device)
        vision_PF_embeds = projector(img_PF_feature)
        del img_PF_feature

        target_length = (max_length - expand_sys_embeds.shape[1] - 3*expand_usual1_embeds.shape[1] - expand_CL_embeds.shape[1] - expand_PF_embeds.shape[1]) // 3

        # 使用插值进行线性缩放
        vision_CF_embeds = vision_CF_embeds.permute(0, 2, 1)  # 形状: (1, 4096, len(input_embedings))
        vision_CL_embeds = vision_CL_embeds.permute(0, 2, 1)
        vision_PF_embeds = vision_PF_embeds.permute(0, 2, 1)
        # 使用线性插值缩放
        scaled_vision_CF_embeds = F.interpolate(vision_CF_embeds, size=target_length, mode='linear', align_corners=False)  # 形状: (1,4096,1530)
        del vision_CF_embeds
        scaled_vision_CL_embeds = F.interpolate(vision_CL_embeds, size=target_length, mode='linear', align_corners=False)  # 形状: (1,4096,1530)
        del vision_CL_embeds
        scaled_vision_PF_embeds = F.interpolate(vision_PF_embeds, size=target_length, mode='linear', align_corners=False)  # 形状: (1,4096,1530)
        del vision_PF_embeds
        # 恢复原始形状
        vision_CF_embeds = scaled_vision_CF_embeds.permute(0, 2, 1)  # 形状: (1,1530, 4096)
        del scaled_vision_CF_embeds
        vision_CL_embeds = scaled_vision_CL_embeds.permute(0, 2, 1)  # 形状: (1,1530, 4096)
        del scaled_vision_CL_embeds 
        vision_PF_embeds = scaled_vision_PF_embeds.permute(0, 2, 1)  # 形状: (1,1530, 4096)
        del scaled_vision_PF_embeds 
        
        input_embeddings = torch.cat([expand_sys_embeds,vision_CF_embeds.to(model.device)],dim=1)
        del expand_sys_embeds
        del vision_CF_embeds
        input_embeddings = torch.cat([input_embeddings,expand_usual1_embeds],dim=1)

        input_embeddings = torch.cat([input_embeddings,expand_CL_embeds],dim=1)
        input_embeddings = torch.cat([input_embeddings,vision_CL_embeds.to(model.device)],dim=1)
        del expand_CL_embeds
        del vision_CL_embeds
        input_embeddings = torch.cat([input_embeddings,expand_usual1_embeds],dim=1)

        input_embeddings = torch.cat([input_embeddings,expand_PF_embeds],dim=1)
        input_embeddings = torch.cat([input_embeddings,vision_PF_embeds.to(model.device)],dim=1)
        del expand_PF_embeds
        del vision_PF_embeds
        input_embeddings = torch.cat([input_embeddings,expand_usual1_embeds],dim=1)
        del expand_usual1_embeds

    else:
        #CF
        input_embeddings = torch.cat([expand_sys_embeds,vision_CF_embeds.to(model.device)],dim=1)
        del expand_sys_embeds
        input_embeddings = torch.cat([input_embeddings,expand_usual1_embeds],dim=1)
        #CL
        if img_CL_feature != ('None',):
            expand_CL_embeds = TrainingConfig.prompt_embed['C_lateral_embed'].expand(vision_CF_embeds.size(0),-1,-1)
            img_CL_feature = img_CL_feature.to(next(projector.parameters()).device)
            vision_CL_embeds = projector(img_CL_feature)
            del img_CL_feature
            input_embeddings = torch.cat([input_embeddings,expand_CL_embeds],dim=1)
            input_embeddings = torch.cat([input_embeddings,vision_CL_embeds.to(model.device)],dim=1)
            del expand_CL_embeds
            del vision_CL_embeds
            input_embeddings = torch.cat([input_embeddings,expand_usual1_embeds],dim=1)
        #PF
        if img_PF_feature != ('None',):
            expand_PF_embeds = TrainingConfig.prompt_embed['P_frontal_embeds'].expand(vision_CF_embeds.size(0),-1,-1)
            img_PF_feature = img_PF_feature.to(next(projector.parameters()).device)
            vision_PF_embeds = projector(img_PF_feature)
            del img_PF_feature
            input_embeddings = torch.cat([input_embeddings,expand_PF_embeds],dim=1)
            input_embeddings = torch.cat([input_embeddings,vision_PF_embeds.to(model.device)],dim=1)
            del expand_PF_embeds
            del vision_PF_embeds
            input_embeddings = torch.cat([input_embeddings,expand_usual1_embeds],dim=1)
            del expand_usual1_embeds
            del vision_CF_embeds

    att_mask = F.pad(TrainingConfig.attention_mask['P_Re_att'], (input_embeddings.shape[1],0), value=1)
    #结合pf
    input_embeddings = torch.cat([input_embeddings,expand_pf_embeds],dim=1)
    input_embeddings = torch.cat([input_embeddings,pf.to(model.device)],dim=1)
    del expand_pf_embeds
    del pf
    #结合用户提示以及indication
    input_embeddings = torch.cat([input_embeddings,expand_user_embeds],dim=1)
    input_embeddings = torch.cat([input_embeddings,ind.to(model.device)],dim=1)
    del expand_user_embeds
    del ind
    #结合technique
    input_embeddings = torch.cat([input_embeddings,expand_Tech_embeds],dim=1)
    input_embeddings = torch.cat([input_embeddings,tech.to(model.device)],dim=1)
    del expand_Tech_embeds
    del tech
    #结合comparison
    input_embeddings = torch.cat([input_embeddings,expand_Comp_embeds],dim=1)
    input_embeddings = torch.cat([input_embeddings,comp.to(model.device)],dim=1)
    del expand_Comp_embeds
    del comp

    components_att = [
        att_mask,
        pf_att,
        TrainingConfig.attention_mask['user_att'],
        ind_att,
        TrainingConfig.attention_mask['Tech_att'],
        tech_att,
        TrainingConfig.attention_mask['Comp_att'],
        comp_att,
    ]

    #将labels以及attention进行改变
    att_mask = torch.cat(components_att,dim=-1)
    torch.cuda.empty_cache() 

    result_dict['input_embeding'] = input_embeddings
    result_dict['attention'] = att_mask
    return result_dict
#____________________________________________________________________________________________________________________________________________________________________fuse1_4096

def fuse23_test(TrainingConfig,model,fusion,projector,img_CF_feature,img_PF_feature,img_CL_feature,ind_att,ind,tech_att,tech,comp_att,comp,pf_att,pf):
    result_dict = {}
    views = {}

    #扩展提示嵌入以匹配视觉特征
    expand_sys_embeds = TrainingConfig.prompt_embed['sys_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_user_embeds = TrainingConfig.prompt_embed['user_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_usual1_embeds = TrainingConfig.prompt_embed['usual1_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_pf_embeds = TrainingConfig.prompt_embed['P_Re_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Tech_embeds = TrainingConfig.prompt_embed['Tech_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Comp_embeds = TrainingConfig.prompt_embed['Comp_embeds'].expand(TrainingConfig.batch_size,-1,-1)

    # 确定输入类型
    if isinstance(img_PF_feature, torch.Tensor) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL + PF'
        views['cf'] = img_CF_feature.to(next(fusion.parameters()).device)    #[1,1369,768]
        views['cl'] = img_CL_feature.to(next(fusion.parameters()).device)    #[1,1369,768]
        views['pf'] = img_PF_feature.to(next(fusion.parameters()).device)    #[1,1369,768]
    elif isinstance(img_PF_feature, torch.Tensor) and (not isinstance(img_CL_feature, torch.Tensor)):
        input_type = 'CF + PF'
        views['cf'] = img_CF_feature.to(next(fusion.parameters()).device)    #[1,1369,768]
        views['pf'] = img_PF_feature.to(next(fusion.parameters()).device)    #[1,1369,768]
    elif (not isinstance(img_PF_feature, torch.Tensor)) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL'
        views['cf'] = img_CF_feature.to(next(fusion.parameters()).device)    #[1,1369,768]
        views['cl'] = img_CL_feature.to(next(fusion.parameters()).device)     #[1,1369,768]
    else:
        input_type = 'CF only'
        views['cf'] = img_CF_feature.to(next(fusion.parameters()).device)    #[1,1369,768]
    result_dict['input_type'] = input_type

    #判断注意力方法
    if TrainingConfig.fusion_type == FusionType.SELF_ATTN:
        #拼接图像
        temp_views = []
        for key in views.keys():
            temp_views.append(views[key])
        image_features = torch.cat(temp_views,dim=1) if len(temp_views) > 1 else temp_views[0]
        #注意力机制压缩融合输入图像信息
        image_features = fusion(image_features)
    elif TrainingConfig.fusion_type == FusionType.CROSS_ATTN:
        image_features = fusion(features_dict=views)
    
    #投影到文本空间
    vision_embeds = projector(image_features.to(next(projector.parameters()).device))
    vision_embeds = vision_embeds.to(model.device)
    if input_type == 'CF only':
        input_embeddings = torch.cat([expand_sys_embeds,vision_embeds,expand_usual1_embeds],dim=1)
        att_mask = F.pad(TrainingConfig.attention_mask['sys_att'], (0,vision_embeds.shape[1]), value=1)
        att_mask = torch.cat([att_mask, TrainingConfig.attention_mask['usual1_att']], dim=-1)
        expand_usual1_embeds = expand_usual1_embeds.detach()

    elif input_type == 'CF + CL + PF':
        expand_sys1_embeds = TrainingConfig.prompt_embed['sys1_embeds'].expand(TrainingConfig.batch_size,-1,-1)
        input_embeddings = torch.cat([expand_sys1_embeds,vision_embeds,expand_usual1_embeds],dim=1)
        att_mask = F.pad(TrainingConfig.attention_mask['sys1_att'], (0,vision_embeds.shape[1]), value=1)
        att_mask = torch.cat([att_mask, TrainingConfig.attention_mask['usual1_att']], dim=-1)
        expand_sys1_embeds = expand_sys1_embeds.detach()

    elif input_type == 'CF + PF':
        expand_sys2_embeds = TrainingConfig.prompt_embed['sys2_embeds'].expand(TrainingConfig.batch_size,-1,-1)
        input_embeddings = torch.cat([expand_sys2_embeds,vision_embeds,expand_usual1_embeds],dim=1)
        att_mask = F.pad(TrainingConfig.attention_mask['sys2_att'], (0,vision_embeds.shape[1]), value=1)
        att_mask = torch.cat([att_mask, TrainingConfig.attention_mask['usual1_att']], dim=-1)
        expand_sys2_embeds = expand_sys2_embeds.detach()

    else:
        expand_sys3_embeds = TrainingConfig.prompt_embed['sys3_embeds'].expand(TrainingConfig.batch_size,-1,-1)
        input_embeddings = torch.cat([expand_sys3_embeds,vision_embeds,expand_usual1_embeds],dim=1)
        att_mask = F.pad(TrainingConfig.attention_mask['sys3_att'], (0,vision_embeds.shape[1]), value=1)
        att_mask = torch.cat([att_mask, TrainingConfig.attention_mask['usual1_att']], dim=-1)
        expand_sys3_embeds = expand_sys3_embeds.detach()
        
    # print(input_type)
    pf = pf.to(model.device).detach()
    ind = ind.to(model.device).detach()
    tech = tech.to(model.device).detach()
    comp = comp.to(model.device).detach()
    components = [
        input_embeddings,
        expand_pf_embeds,
        pf,
        expand_user_embeds,
        ind,
        expand_Tech_embeds,
        tech,
        expand_Comp_embeds,
        comp,
    ]
    components_att = [
        att_mask,
        TrainingConfig.attention_mask['P_Re_att'],
        pf_att,
        TrainingConfig.attention_mask['user_att'],
        ind_att,
        TrainingConfig.attention_mask['Tech_att'],
        tech_att,
        TrainingConfig.attention_mask['Comp_att'],
        comp_att,
    ]
    input_embeddings = torch.cat(components, dim=1)
    att_mask = torch.cat(components_att,dim=-1)

    # 手动释放缓存 
    torch.cuda.empty_cache()


    result_dict['input_embeding'] = input_embeddings
    result_dict['attention'] = att_mask
    return result_dict
#____________________________________________________________________________________________________________________________________________________________________fuse2 3
def fuse4_test(TrainingConfig,model,fusion,projector,img_CF_feature,img_PF_feature,img_CL_feature,ind_att,ind,tech_att,tech,comp_att,comp,pf_att,pf):
    result_dict = {}
    views = {}

    #扩展提示嵌入以匹配视觉特征
    expand_sys_embeds = TrainingConfig.prompt_embed['sys_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_user_embeds = TrainingConfig.prompt_embed['user_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_usual1_embeds = TrainingConfig.prompt_embed['usual1_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_pf_embeds = TrainingConfig.prompt_embed['P_Re_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Tech_embeds = TrainingConfig.prompt_embed['Tech_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Comp_embeds = TrainingConfig.prompt_embed['Comp_embeds'].expand(TrainingConfig.batch_size,-1,-1)

    # 确定输入类型
    if isinstance(img_PF_feature, torch.Tensor) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL + PF'
        views['cf'] = img_CF_feature    #[1,1369,768]
        views['cl'] = img_CL_feature    #[1,1369,768]
        views['pf'] = img_PF_feature    #[1,1369,768]
    elif isinstance(img_PF_feature, torch.Tensor) and (not isinstance(img_CL_feature, torch.Tensor)):
        input_type = 'CF + PF'
        views['cf'] = img_CF_feature    #[1,1369,768]
        views['pf'] = img_PF_feature    #[1,1369,768]
    elif (not isinstance(img_PF_feature, torch.Tensor)) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL'
        views['cf'] = img_CF_feature    #[1,1369,768]
        views['cl'] = img_CL_feature     #[1,1369,768]
    else:
        input_type = 'CF only'
        views['cf'] = img_CF_feature    #[1,1369,768]
    result_dict['input_type'] = input_type

    #注意力机制融合信息
    fusion_result = fusion(features_dict=views)
    
    #投影到文本空间
    vision_embeds = projector(fusion_result.to(next(projector.parameters()).device))
    vision_embeds = vision_embeds.to(model.device)
    if input_type == 'CF only':
        input_embeddings = torch.cat([expand_sys_embeds,vision_embeds,expand_usual1_embeds],dim=1)
        att_mask = F.pad(TrainingConfig.attention_mask['sys_att'], (0,vision_embeds.shape[1]), value=1)
        att_mask = torch.cat([att_mask, TrainingConfig.attention_mask['usual1_att']], dim=-1)
        expand_usual1_embeds = expand_usual1_embeds.detach()

    elif input_type == 'CF + CL + PF':
        expand_sys1_embeds = TrainingConfig.prompt_embed['sys1_embeds'].expand(TrainingConfig.batch_size,-1,-1)
        input_embeddings = torch.cat([expand_sys1_embeds,vision_embeds,expand_usual1_embeds],dim=1)
        att_mask = F.pad(TrainingConfig.attention_mask['sys1_att'], (0,vision_embeds.shape[1]), value=1)
        att_mask = torch.cat([att_mask, TrainingConfig.attention_mask['usual1_att']], dim=-1)
        expand_sys1_embeds = expand_sys1_embeds.detach()

    elif input_type == 'CF + PF':
        expand_sys2_embeds = TrainingConfig.prompt_embed['sys2_embeds'].expand(TrainingConfig.batch_size,-1,-1)
        input_embeddings = torch.cat([expand_sys2_embeds,vision_embeds,expand_usual1_embeds],dim=1)
        att_mask = F.pad(TrainingConfig.attention_mask['sys2_att'], (0,vision_embeds.shape[1]), value=1)
        att_mask = torch.cat([att_mask, TrainingConfig.attention_mask['usual1_att']], dim=-1)
        expand_sys2_embeds = expand_sys2_embeds.detach()

    else:
        expand_sys3_embeds = TrainingConfig.prompt_embed['sys3_embeds'].expand(TrainingConfig.batch_size,-1,-1)
        input_embeddings = torch.cat([expand_sys3_embeds,vision_embeds,expand_usual1_embeds],dim=1)
        att_mask = F.pad(TrainingConfig.attention_mask['sys3_att'], (0,vision_embeds.shape[1]), value=1)
        att_mask = torch.cat([att_mask, TrainingConfig.attention_mask['usual1_att']], dim=-1)
        expand_sys3_embeds = expand_sys3_embeds.detach()
        
    # print(input_type)
    pf = pf.to(model.device).detach()
    ind = ind.to(model.device).detach()
    tech = tech.to(model.device).detach()
    comp = comp.to(model.device).detach()
    components = [
        input_embeddings,
        expand_pf_embeds,
        pf,
        expand_user_embeds,
        ind,
        expand_Tech_embeds,
        tech,
        expand_Comp_embeds,
        comp,
    ]
    components_att = [
        att_mask,
        TrainingConfig.attention_mask['P_Re_att'],
        pf_att,
        TrainingConfig.attention_mask['user_att'],
        ind_att,
        TrainingConfig.attention_mask['Tech_att'],
        tech_att,
        TrainingConfig.attention_mask['Comp_att'],
        comp_att,
    ]
    
    input_embeddings = torch.cat(components, dim=1)
    att_mask = torch.cat(components_att,dim=-1)

    # 手动释放缓存 
    torch.cuda.empty_cache()


    result_dict['input_embeding'] = input_embeddings
    result_dict['attention'] = att_mask
    return result_dict
#____________________________________________________________________________________________________________________________________________________________________fuse4
def fuse4_test_think(TrainingConfig,model,fusion,projector,img_CF_feature,img_PF_feature,img_CL_feature,ind_att,ind,tech_att,tech,comp_att,comp,pf_att,pf):
    result_dict = {}
    views = {}

    #扩展提示嵌入以匹配视觉特征
    expand_sys_embeds = TrainingConfig.prompt_embed['sys_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_user_embeds = TrainingConfig.prompt_embed['user_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_usual1_embeds = TrainingConfig.prompt_embed['usual1_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_pf_embeds = TrainingConfig.prompt_embed['P_Re_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Tech_embeds = TrainingConfig.prompt_embed['Tech_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Comp_embeds = TrainingConfig.prompt_embed['Comp_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_thk_embeds = TrainingConfig.prompt_embed['thought_embeds'].expand(TrainingConfig.batch_size,-1,-1)

    # 确定输入类型
    if isinstance(img_PF_feature, torch.Tensor) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL + PF'
        views['cf'] = img_CF_feature    #[1,1369,768]
        views['cl'] = img_CL_feature    #[1,1369,768]
        views['pf'] = img_PF_feature    #[1,1369,768]
    elif isinstance(img_PF_feature, torch.Tensor) and (not isinstance(img_CL_feature, torch.Tensor)):
        input_type = 'CF + PF'
        views['cf'] = img_CF_feature    #[1,1369,768]
        views['pf'] = img_PF_feature    #[1,1369,768]
    elif (not isinstance(img_PF_feature, torch.Tensor)) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL'
        views['cf'] = img_CF_feature    #[1,1369,768]
        views['cl'] = img_CL_feature     #[1,1369,768]
    else:
        input_type = 'CF only'
        views['cf'] = img_CF_feature    #[1,1369,768]
    result_dict['input_type'] = input_type

    #注意力机制融合信息
    fusion_result = fusion(features_dict=views)
    
    #投影到文本空间
    vision_embeds = projector(fusion_result.to(next(projector.parameters()).device))
    vision_embeds = vision_embeds.to(model.device)
    if input_type == 'CF only':
        input_embeddings = torch.cat([expand_sys_embeds,vision_embeds,expand_usual1_embeds],dim=1)
        att_mask = F.pad(TrainingConfig.attention_mask['sys_att'], (0,vision_embeds.shape[1]), value=1)
        att_mask = torch.cat([att_mask, TrainingConfig.attention_mask['usual1_att']], dim=-1)
        expand_usual1_embeds = expand_usual1_embeds.detach()

    elif input_type == 'CF + CL + PF':
        expand_sys1_embeds = TrainingConfig.prompt_embed['sys1_embeds'].expand(TrainingConfig.batch_size,-1,-1)
        input_embeddings = torch.cat([expand_sys1_embeds,vision_embeds,expand_usual1_embeds],dim=1)
        att_mask = F.pad(TrainingConfig.attention_mask['sys1_att'], (0,vision_embeds.shape[1]), value=1)
        att_mask = torch.cat([att_mask, TrainingConfig.attention_mask['usual1_att']], dim=-1)
        expand_sys1_embeds = expand_sys1_embeds.detach()

    elif input_type == 'CF + PF':
        expand_sys2_embeds = TrainingConfig.prompt_embed['sys2_embeds'].expand(TrainingConfig.batch_size,-1,-1)
        input_embeddings = torch.cat([expand_sys2_embeds,vision_embeds,expand_usual1_embeds],dim=1)
        att_mask = F.pad(TrainingConfig.attention_mask['sys2_att'], (0,vision_embeds.shape[1]), value=1)
        att_mask = torch.cat([att_mask, TrainingConfig.attention_mask['usual1_att']], dim=-1)
        expand_sys2_embeds = expand_sys2_embeds.detach()

    else:
        expand_sys3_embeds = TrainingConfig.prompt_embed['sys3_embeds'].expand(TrainingConfig.batch_size,-1,-1)
        input_embeddings = torch.cat([expand_sys3_embeds,vision_embeds,expand_usual1_embeds],dim=1)
        att_mask = F.pad(TrainingConfig.attention_mask['sys3_att'], (0,vision_embeds.shape[1]), value=1)
        att_mask = torch.cat([att_mask, TrainingConfig.attention_mask['usual1_att']], dim=-1)
        expand_sys3_embeds = expand_sys3_embeds.detach()
        
    # print(input_type)
    pf = pf.to(model.device).detach()
    ind = ind.to(model.device).detach()
    tech = tech.to(model.device).detach()
    comp = comp.to(model.device).detach()
    components = [
        input_embeddings,
        expand_pf_embeds,
        pf,
        expand_user_embeds,
        ind,
        expand_Tech_embeds,
        tech,
        expand_Comp_embeds,
        comp,
        expand_thk_embeds,
    ]
    components_att = [
        att_mask,
        TrainingConfig.attention_mask['P_Re_att'],
        pf_att,
        TrainingConfig.attention_mask['user_att'],
        ind_att,
        TrainingConfig.attention_mask['Tech_att'],
        tech_att,
        TrainingConfig.attention_mask['Comp_att'],
        comp_att,
        TrainingConfig.attention_mask['Thought_att'],
    ]
    input_embeddings = torch.cat(components, dim=1)
    att_mask = torch.cat(components_att,dim=-1)

    # 手动释放缓存 
    torch.cuda.empty_cache()


    result_dict['input_embeding'] = input_embeddings
    result_dict['attention'] = att_mask
    return result_dict
#____________________________________________________________________________________________________________________________________________________________________fuse4_think

def openi(TrainingConfig,model,projector,img_CF_feature,category='all',ind_att=None,ind=None,tech_att=None,tech=None,comp_att=None,comp=None,pf_att=None,pf=None):
    result_dict = {}
    # 获取视觉嵌入-1
    img_CF_feature = img_CF_feature.to(next(projector.parameters()).device)
    # print(img_CF_feature.shape)
    vision_CF_embeds = projector(img_CF_feature)
    # print(vision_CF_embeds.shape)
    del img_CF_feature

    #扩展提示嵌入以匹配视觉特征
    expand_sys_embeds = TrainingConfig.prompt_embed['sys_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_user_embeds = TrainingConfig.prompt_embed['user_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_usual1_embeds = TrainingConfig.prompt_embed['usual1_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_pf_embeds = TrainingConfig.prompt_embed['P_Re_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Tech_embeds = TrainingConfig.prompt_embed['Tech_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Comp_embeds = TrainingConfig.prompt_embed['Comp_embeds'].expand(TrainingConfig.batch_size,-1,-1)

    #注意力
    att_mask = F.pad(TrainingConfig.attention_mask['sys_att'], (0,vision_CF_embeds.shape[1]), value=1)
    att_mask = torch.cat([att_mask, TrainingConfig.attention_mask['usual1_att']], dim=-1)
    #结合系统提示以及CF,CL.PF
    #CF
    input_embeddings = torch.cat([expand_sys_embeds,vision_CF_embeds.to(model.device)],dim=1)
    input_embeddings = torch.cat([input_embeddings,expand_usual1_embeds],dim=1)
    del expand_sys_embeds
    del vision_CF_embeds
    if category == "all":
        #结合pf
        input_embeddings = torch.cat([input_embeddings,expand_pf_embeds],dim=1)
        input_embeddings = torch.cat([input_embeddings,pf.to(model.device)],dim=1)
        del expand_pf_embeds
        del pf
        #结合用户提示以及indication
        input_embeddings = torch.cat([input_embeddings,expand_user_embeds],dim=1)
        input_embeddings = torch.cat([input_embeddings,ind.to(model.device)],dim=1)
        del expand_user_embeds
        del ind
        #结合technique
        input_embeddings = torch.cat([input_embeddings,expand_Tech_embeds],dim=1)
        input_embeddings = torch.cat([input_embeddings,tech.to(model.device)],dim=1)
        del expand_Tech_embeds
        del tech
        #结合comparison
        input_embeddings = torch.cat([input_embeddings,expand_Comp_embeds],dim=1)
        input_embeddings = torch.cat([input_embeddings,comp.to(model.device)],dim=1)
        del expand_Comp_embeds
        del comp

        components_att = [
            att_mask,
            pf_att,
            TrainingConfig.attention_mask['user_att'],
            ind_att,
            TrainingConfig.attention_mask['Tech_att'],
            tech_att,
            TrainingConfig.attention_mask['Comp_att'],
            comp_att,
        ]
        att_mask = torch.cat(components_att,dim=-1)
    if category == 'indication':
        #结合用户提示以及indication
        input_embeddings = torch.cat([input_embeddings,expand_user_embeds],dim=1)
        input_embeddings = torch.cat([input_embeddings,ind.to(model.device)],dim=1)
        del expand_user_embeds
        del ind
        att_mask = torch.cat([att_mask,TrainingConfig.attention_mask['user_att'],ind_att],dim=1)
    # print(labels[:,-140:])
    # 手动释放缓存ssssssssss
    torch.cuda.empty_cache() 

    result_dict['input_embeding'] = input_embeddings
    result_dict['attention'] = att_mask
    return result_dict
#____________________________________________________________________________________________________________________________________________________________________openi




def fuse1_Qwen(TrainingConfig,model,LLMs_token,projector,img_CF_feature,img_PF_feature,img_CL_feature,labels,att,embeds,ind_att,ind,tech_att,tech,comp_att,comp,pf_att,pf):
    result_dict = {}
    # 获取视觉嵌入-1
    img_CF_feature = img_CF_feature.to(next(projector.parameters()).device)
    # print(img_CF_feature.shape)
    vision_CF_embeds = projector(img_CF_feature)
    # print(vision_CF_embeds.shape)
    del img_CF_feature

    #扩展提示嵌入以匹配视觉特征
    expand_usual1_img_embeds = TrainingConfig.prompt_embed['usual1_img_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_sys_embeds = TrainingConfig.prompt_embed['sys_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_user_embeds = TrainingConfig.prompt_embed['user_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_usual1_embeds = TrainingConfig.prompt_embed['usual1_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_pf_embeds = TrainingConfig.prompt_embed['P_Re_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Tech_embeds = TrainingConfig.prompt_embed['Tech_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Comp_embeds = TrainingConfig.prompt_embed['Comp_embeds'].expand(TrainingConfig.batch_size,-1,-1)

    # 确定输入类型
    if isinstance(img_PF_feature, torch.Tensor) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL + PF'
    elif isinstance(img_PF_feature, torch.Tensor) and (not isinstance(img_CL_feature, torch.Tensor)):
        input_type = 'CF + PF'
    elif (not isinstance(img_PF_feature, torch.Tensor)) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL'
    else:
        input_type = 'CF only'

    result_dict['input_type'] = input_type
    #结合系统提示以及CF,CL.PF
    #CF
    input_embeddings = torch.cat([expand_sys_embeds,vision_CF_embeds.to(model.device)],dim=1)
    del expand_sys_embeds
    input_embeddings = torch.cat([input_embeddings,expand_usual1_img_embeds],dim=1)
    #CL
    if img_CL_feature != ('None',):
        expand_CL_embeds = TrainingConfig.prompt_embed['C_lateral_embed'].expand(vision_CF_embeds.size(0),-1,-1)
        img_CL_feature = img_CL_feature.to(next(projector.parameters()).device)
        vision_CL_embeds = projector(img_CL_feature)
        del img_CL_feature
        input_embeddings = torch.cat([input_embeddings,expand_CL_embeds],dim=1)
        input_embeddings = torch.cat([input_embeddings,vision_CL_embeds.to(model.device)],dim=1)
        del expand_CL_embeds
        del vision_CL_embeds
        input_embeddings = torch.cat([input_embeddings,expand_usual1_img_embeds],dim=1)
    #PF
    if img_PF_feature != ('None',):
        expand_PF_embeds = TrainingConfig.prompt_embed['P_frontal_embeds'].expand(vision_CF_embeds.size(0),-1,-1)
        img_PF_feature = img_PF_feature.to(next(projector.parameters()).device)
        vision_PF_embeds = projector(img_PF_feature)
        del img_PF_feature
        input_embeddings = torch.cat([input_embeddings,expand_PF_embeds],dim=1)
        input_embeddings = torch.cat([input_embeddings,vision_PF_embeds.to(model.device)],dim=1)
        del expand_PF_embeds
        del vision_PF_embeds
        input_embeddings = torch.cat([input_embeddings,expand_usual1_img_embeds],dim=1)
        del expand_usual1_embeds
        del vision_CF_embeds

    # print(f"缩放前的输入维度:{input_embeddings.shape}")
    # 定义目标长度
    target_length = 1712  
    # target_length = input_embeddings.shape[1] / 2.4
    if target_length < input_embeddings.shape[1]:
        # 使用插值进行线性缩放
        input_embeddings = input_embeddings.permute(0, 2, 1)  # 形状: (1, 4096, len(input_embedings))
        # 使用线性插值缩放
        scaled_input_embeddings = F.interpolate(input_embeddings, size=target_length, mode='linear', align_corners=False)  # 形状: (1,4096,1530)
        del input_embeddings
        # 恢复原始形状
        input_embeddings = scaled_input_embeddings.permute(0, 2, 1)  # 形状: (1,1530, 4096)
        del scaled_input_embeddings
    # print(f"缩放后的输入维度:{input_embeddings.shape}")
    att_mask = F.pad(TrainingConfig.attention_mask['P_Re_att'], (input_embeddings.shape[1],0), value=1)
    #结合pf
    input_embeddings = torch.cat([input_embeddings,expand_pf_embeds],dim=1)
    input_embeddings = torch.cat([input_embeddings,pf.to(model.device)],dim=1)
    del expand_pf_embeds
    del pf
    #结合用户提示以及indication
    input_embeddings = torch.cat([input_embeddings,expand_user_embeds],dim=1)
    input_embeddings = torch.cat([input_embeddings,ind.to(model.device)],dim=1)
    del expand_user_embeds
    del ind
    #结合technique
    input_embeddings = torch.cat([input_embeddings,expand_Tech_embeds],dim=1)
    input_embeddings = torch.cat([input_embeddings,tech.to(model.device)],dim=1)
    del expand_Tech_embeds
    del tech
    #结合comparison
    input_embeddings = torch.cat([input_embeddings,expand_Comp_embeds],dim=1)
    input_embeddings = torch.cat([input_embeddings,comp.to(model.device)],dim=1)
    del expand_Comp_embeds
    del comp

    components_att = [
        att_mask,
        pf_att,
        TrainingConfig.attention_mask['user_att'],
        ind_att,
        TrainingConfig.attention_mask['Tech_att'],
        tech_att,
        TrainingConfig.attention_mask['Comp_att'],
        comp_att,
        att
    ]

    #将labels以及attention进行改变
    att_mask = torch.cat(components_att,dim=-1)
    labels = F.pad(labels, (input_embeddings.shape[1],0), value=-100)
    labels = labels.masked_fill(labels == LLMs_token.pad_token_id, -100)
    labels = labels.to(model.device, dtype=torch.int64)
    #将目标文本进行嵌入
    input_embeddings = torch.cat((input_embeddings,embeds.to(model.device)),dim=1)
    del embeds
    print(f"input_embeddings的维度为{input_embeddings.shape},labels的维度为{labels.shape},att的维度为{att_mask.shape}")
    # print(labels[:,-140:])
    # 手动释放缓存
    torch.cuda.empty_cache() 

    result_dict['input_embeding'] = input_embeddings
    result_dict['attention'] = att_mask
    result_dict['labels'] = labels
    return result_dict

def  fuse4_Qwen(TrainingConfig,model,LLMs_token,fusion,projector,img_CF_feature,img_PF_feature,img_CL_feature,labels,att,embeds,ind_att,ind,tech_att,tech,comp_att,comp,pf_att,pf):
    result_dict = {}
    views = {}
    # print(f"Qwen的嵌入形状是{embeds.shape},{ind.shape}")

    #扩展提示嵌入以匹配视觉特征
    expand_usual1_img_embeds = TrainingConfig.prompt_embed['usual1_img_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_sys_embeds = TrainingConfig.prompt_embed['sys_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_user_embeds = TrainingConfig.prompt_embed['user_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_pf_embeds = TrainingConfig.prompt_embed['P_Re_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Tech_embeds = TrainingConfig.prompt_embed['Tech_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Comp_embeds = TrainingConfig.prompt_embed['Comp_embeds'].expand(TrainingConfig.batch_size,-1,-1)

    # 确定输入类型
    if isinstance(img_PF_feature, torch.Tensor) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL + PF'
        views['cf'] = img_CF_feature    #[1,1369,768]
        views['cl'] = img_CL_feature    #[1,1369,768]
        views['pf'] = img_PF_feature    #[1,1369,768]
    elif isinstance(img_PF_feature, torch.Tensor) and (not isinstance(img_CL_feature, torch.Tensor)):
        input_type = 'CF + PF'
        views['cf'] = img_CF_feature    #[1,1369,768]
        views['pf'] = img_PF_feature    #[1,1369,768]
    elif (not isinstance(img_PF_feature, torch.Tensor)) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL'
        views['cf'] = img_CF_feature    #[1,1369,768]
        views['cl'] = img_CL_feature     #[1,1369,768]
    else:
        input_type = 'CF only'
        views['cf'] = img_CF_feature    #[1,1369,768]
    result_dict['input_type'] = input_type

    #注意力机制融合信息
    fusion_result = fusion(features_dict=views)
    
    #投影到文本空间
    vision_embeds = projector(fusion_result.to(next(projector.parameters()).device))
    vision_embeds = vision_embeds.to(model.device)
    if input_type == 'CF only':
        input_embeddings = torch.cat([expand_sys_embeds,vision_embeds,expand_usual1_img_embeds],dim=1)
        att_mask = F.pad(TrainingConfig.attention_mask['sys_att'], (0,vision_embeds.shape[1]), value=1)
        att_mask = torch.cat([att_mask, TrainingConfig.attention_mask['usual1_att']], dim=-1)
        expand_usual1_img_embeds = expand_usual1_img_embeds.detach()

    elif input_type == 'CF + CL + PF':
        expand_sys1_embeds = TrainingConfig.prompt_embed['sys1_embeds'].expand(TrainingConfig.batch_size,-1,-1)
        input_embeddings = torch.cat([expand_sys1_embeds,vision_embeds,expand_usual1_img_embeds],dim=1)
        att_mask = F.pad(TrainingConfig.attention_mask['sys1_att'], (0,vision_embeds.shape[1]), value=1)
        att_mask = torch.cat([att_mask, TrainingConfig.attention_mask['usual1_att']], dim=-1)
        expand_sys1_embeds = expand_sys1_embeds.detach()

    elif input_type == 'CF + PF':
        expand_sys2_embeds = TrainingConfig.prompt_embed['sys2_embeds'].expand(TrainingConfig.batch_size,-1,-1)
        input_embeddings = torch.cat([expand_sys2_embeds,vision_embeds,expand_usual1_img_embeds],dim=1)
        att_mask = F.pad(TrainingConfig.attention_mask['sys2_att'], (0,vision_embeds.shape[1]), value=1)
        att_mask = torch.cat([att_mask, TrainingConfig.attention_mask['usual1_att']], dim=-1)
        expand_sys2_embeds = expand_sys2_embeds.detach()

    else:
        expand_sys3_embeds = TrainingConfig.prompt_embed['sys3_embeds'].expand(TrainingConfig.batch_size,-1,-1)
        input_embeddings = torch.cat([expand_sys3_embeds,vision_embeds,expand_usual1_img_embeds],dim=1)
        att_mask = F.pad(TrainingConfig.attention_mask['sys3_att'], (0,vision_embeds.shape[1]), value=1)
        att_mask = torch.cat([att_mask, TrainingConfig.attention_mask['usual1_att']], dim=-1)
        expand_sys3_embeds = expand_sys3_embeds.detach()
        
    # print(input_type)
    pf = pf.to(model.device).detach()
    ind = ind.to(model.device).detach()
    tech = tech.to(model.device).detach()
    comp = comp.to(model.device).detach()

    components = [
        input_embeddings,
        expand_pf_embeds,
        pf,
        expand_user_embeds,
        ind,
        expand_Tech_embeds,
        tech,
        expand_Comp_embeds,
        comp,
    ]
    # print(P_Re_att.shape,pf_att.shape,user_att.shape,ind_att.shape,Tech_att.shape,tech_att.shape,Comp_att.shape,comp_att.shape, att.shape)
    components_att = [
        att_mask,
        TrainingConfig.attention_mask['P_Re_att'],
        pf_att,
        TrainingConfig.attention_mask['user_att'],
        ind_att,
        TrainingConfig.attention_mask['Tech_att'],
        tech_att,
        TrainingConfig.attention_mask['Comp_att'],
        comp_att,
        att
    ]
    # print(f"注意力掩码形状为{att_mask.shape}")
    input_embeddings = torch.cat(components, dim=1)
    att_mask = torch.cat(components_att,dim=-1)
    # print(f"注意力掩码形状为{att_mask[:,1416:1500]}")

    #将labels进行改变
    # print(LLMs_token.pad_token_id)
    labels = labels.masked_fill(labels == LLMs_token.pad_token_id, -100)
    labels = F.pad(labels, (input_embeddings.shape[1],0), value=-100)
    labels = labels.to(model.device, dtype=torch.int64)
    #将目标文本进行嵌入
    input_embeddings = torch.cat((input_embeddings,embeds.to(model.device)),dim=1)
    embeds = embeds.detach()

    # 手动释放缓存 
    torch.cuda.empty_cache()


    result_dict['input_embeding'] = input_embeddings
    result_dict['attention'] = att_mask
    result_dict['labels'] = labels
    return result_dict

def fuse1_Qwen_test(TrainingConfig,model,projector,img_CF_feature,img_PF_feature,img_CL_feature,ind_att,ind,tech_att,tech,comp_att,comp,pf_att,pf):
    result_dict = {}
    # 获取视觉嵌入-1
    img_CF_feature = img_CF_feature.to(next(projector.parameters()).device)
    # print(img_CF_feature.shape)
    vision_CF_embeds = projector(img_CF_feature)
    # print(vision_CF_embeds.shape)
    del img_CF_feature

    #扩展提示嵌入以匹配视觉特征
    expand_usual1_img_embeds = TrainingConfig.prompt_embed['usual1_img_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_sys_embeds = TrainingConfig.prompt_embed['sys_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_user_embeds = TrainingConfig.prompt_embed['user_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_pf_embeds = TrainingConfig.prompt_embed['P_Re_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Tech_embeds = TrainingConfig.prompt_embed['Tech_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Comp_embeds = TrainingConfig.prompt_embed['Comp_embeds'].expand(TrainingConfig.batch_size,-1,-1)

    # 确定输入类型
    if isinstance(img_PF_feature, torch.Tensor) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL + PF'
    elif isinstance(img_PF_feature, torch.Tensor) and (not isinstance(img_CL_feature, torch.Tensor)):
        input_type = 'CF + PF'
    elif (not isinstance(img_PF_feature, torch.Tensor)) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL'
    else:
        input_type = 'CF only'

    result_dict['input_type'] = input_type
    #结合系统提示以及CF,CL.PF
    #CF
    input_embeddings = torch.cat([expand_sys_embeds,vision_CF_embeds.to(model.device)],dim=1)
    del expand_sys_embeds
    input_embeddings = torch.cat([input_embeddings,expand_usual1_img_embeds],dim=1)
    #CL
    if img_CL_feature != ('None',):
        expand_CL_embeds = TrainingConfig.prompt_embed['C_lateral_embed'].expand(vision_CF_embeds.size(0),-1,-1)
        img_CL_feature = img_CL_feature.to(next(projector.parameters()).device)
        vision_CL_embeds = projector(img_CL_feature)
        del img_CL_feature
        input_embeddings = torch.cat([input_embeddings,expand_CL_embeds],dim=1)
        input_embeddings = torch.cat([input_embeddings,vision_CL_embeds.to(model.device)],dim=1)
        del expand_CL_embeds
        del vision_CL_embeds
        input_embeddings = torch.cat([input_embeddings,expand_usual1_img_embeds],dim=1)
    #PF
    if img_PF_feature != ('None',):
        expand_PF_embeds = TrainingConfig.prompt_embed['P_frontal_embeds'].expand(vision_CF_embeds.size(0),-1,-1)
        img_PF_feature = img_PF_feature.to(next(projector.parameters()).device)
        vision_PF_embeds = projector(img_PF_feature)
        del img_PF_feature
        input_embeddings = torch.cat([input_embeddings,expand_PF_embeds],dim=1)
        input_embeddings = torch.cat([input_embeddings,vision_PF_embeds.to(model.device)],dim=1)
        del expand_PF_embeds
        del vision_PF_embeds
        input_embeddings = torch.cat([input_embeddings,expand_usual1_img_embeds],dim=1)
        del expand_usual1_embeds
        del vision_CF_embeds

    # print(f"缩放前的输入维度:{input_embeddings.shape}")
    # 定义目标长度
    target_length = 1712  
    # target_length = input_embeddings.shape[1] / 2.4
    if target_length < input_embeddings.shape[1]:
        # 使用插值进行线性缩放
        input_embeddings = input_embeddings.permute(0, 2, 1)  # 形状: (1, 4096, len(input_embedings))
        # 使用线性插值缩放
        scaled_input_embeddings = F.interpolate(input_embeddings, size=target_length, mode='linear', align_corners=False)  # 形状: (1,4096,1530)
        del input_embeddings
        # 恢复原始形状
        input_embeddings = scaled_input_embeddings.permute(0, 2, 1)  # 形状: (1,1530, 4096)
        del scaled_input_embeddings
    # print(f"缩放后的输入维度:{input_embeddings.shape}")
    att_mask = F.pad(TrainingConfig.attention_mask['P_Re_att'], (input_embeddings.shape[1],0), value=1)
    #结合pf
    input_embeddings = torch.cat([input_embeddings,expand_pf_embeds],dim=1)
    input_embeddings = torch.cat([input_embeddings,pf.to(model.device)],dim=1)
    del expand_pf_embeds
    del pf
    #结合用户提示以及indication
    input_embeddings = torch.cat([input_embeddings,expand_user_embeds],dim=1)
    input_embeddings = torch.cat([input_embeddings,ind.to(model.device)],dim=1)
    del expand_user_embeds
    del ind
    #结合technique
    input_embeddings = torch.cat([input_embeddings,expand_Tech_embeds],dim=1)
    input_embeddings = torch.cat([input_embeddings,tech.to(model.device)],dim=1)
    del expand_Tech_embeds
    del tech
    #结合comparison
    input_embeddings = torch.cat([input_embeddings,expand_Comp_embeds],dim=1)
    input_embeddings = torch.cat([input_embeddings,comp.to(model.device)],dim=1)
    del expand_Comp_embeds
    del comp

    components_att = [
        att_mask,
        pf_att,
        TrainingConfig.attention_mask['user_att'],
        ind_att,
        TrainingConfig.attention_mask['Tech_att'],
        tech_att,
        TrainingConfig.attention_mask['Comp_att'],
        comp_att,
    ]

    #将labels以及attention进行改变
    att_mask = torch.cat(components_att,dim=-1)
    # print(labels[:,-140:])
    # 手动释放缓存
    torch.cuda.empty_cache() 

    result_dict['input_embeding'] = input_embeddings
    result_dict['attention'] = att_mask
    return result_dict

def fuse4_Qwen_test(TrainingConfig,model,fusion,projector,img_CF_feature,img_PF_feature,img_CL_feature,ind_att,ind,tech_att,tech,comp_att,comp,pf_att,pf):
    result_dict = {}
    views = {}

    #扩展提示嵌入以匹配视觉特征
    expand_usual1_img_embeds = TrainingConfig.prompt_embed['usual1_img_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_sys_embeds = TrainingConfig.prompt_embed['sys_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_user_embeds = TrainingConfig.prompt_embed['user_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_usual1_embeds = TrainingConfig.prompt_embed['usual1_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_pf_embeds = TrainingConfig.prompt_embed['P_Re_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Tech_embeds = TrainingConfig.prompt_embed['Tech_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Comp_embeds = TrainingConfig.prompt_embed['Comp_embeds'].expand(TrainingConfig.batch_size,-1,-1)

    # 确定输入类型
    if isinstance(img_PF_feature, torch.Tensor) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL + PF'
        views['cf'] = img_CF_feature    #[1,1369,768]
        views['cl'] = img_CL_feature    #[1,1369,768]
        views['pf'] = img_PF_feature    #[1,1369,768]
    elif isinstance(img_PF_feature, torch.Tensor) and (not isinstance(img_CL_feature, torch.Tensor)):
        input_type = 'CF + PF'
        views['cf'] = img_CF_feature    #[1,1369,768]
        views['pf'] = img_PF_feature    #[1,1369,768]
    elif (not isinstance(img_PF_feature, torch.Tensor)) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL'
        views['cf'] = img_CF_feature    #[1,1369,768]
        views['cl'] = img_CL_feature     #[1,1369,768]
    else:
        input_type = 'CF only'
        views['cf'] = img_CF_feature    #[1,1369,768]
    result_dict['input_type'] = input_type

    #注意力机制融合信息
    fusion_result = fusion(features_dict=views)
    
    #投影到文本空间
    vision_embeds = projector(fusion_result.to(next(projector.parameters()).device))
    vision_embeds = vision_embeds.to(model.device)
    if input_type == 'CF only':
        input_embeddings = torch.cat([expand_sys_embeds,vision_embeds,expand_usual1_img_embeds],dim=1)
        att_mask = F.pad(TrainingConfig.attention_mask['sys_att'], (0,vision_embeds.shape[1]), value=1)
        att_mask = torch.cat([att_mask, TrainingConfig.attention_mask['usual1_att']], dim=-1)
        expand_usual1_embeds = expand_usual1_embeds.detach()

    elif input_type == 'CF + CL + PF':
        expand_sys1_embeds = TrainingConfig.prompt_embed['sys1_embeds'].expand(TrainingConfig.batch_size,-1,-1)
        input_embeddings = torch.cat([expand_sys1_embeds,vision_embeds,expand_usual1_img_embeds],dim=1)
        att_mask = F.pad(TrainingConfig.attention_mask['sys1_att'], (0,vision_embeds.shape[1]), value=1)
        att_mask = torch.cat([att_mask, TrainingConfig.attention_mask['usual1_att']], dim=-1)
        expand_sys1_embeds = expand_sys1_embeds.detach()

    elif input_type == 'CF + PF':
        expand_sys2_embeds = TrainingConfig.prompt_embed['sys2_embeds'].expand(TrainingConfig.batch_size,-1,-1)
        input_embeddings = torch.cat([expand_sys2_embeds,vision_embeds,expand_usual1_img_embeds],dim=1)
        att_mask = F.pad(TrainingConfig.attention_mask['sys2_att'], (0,vision_embeds.shape[1]), value=1)
        att_mask = torch.cat([att_mask, TrainingConfig.attention_mask['usual1_att']], dim=-1)
        expand_sys2_embeds = expand_sys2_embeds.detach()

    else:
        expand_sys3_embeds = TrainingConfig.prompt_embed['sys3_embeds'].expand(TrainingConfig.batch_size,-1,-1)
        input_embeddings = torch.cat([expand_sys3_embeds,vision_embeds,expand_usual1_img_embeds],dim=1)
        att_mask = F.pad(TrainingConfig.attention_mask['sys3_att'], (0,vision_embeds.shape[1]), value=1)
        att_mask = torch.cat([att_mask, TrainingConfig.attention_mask['usual1_att']], dim=-1)
        expand_sys3_embeds = expand_sys3_embeds.detach()
        
    # print(input_type)
    pf = pf.to(model.device).detach()
    ind = ind.to(model.device).detach()
    tech = tech.to(model.device).detach()
    comp = comp.to(model.device).detach()
    components = [
        input_embeddings,
        expand_pf_embeds,
        pf,
        expand_user_embeds,
        ind,
        expand_Tech_embeds,
        tech,
        expand_Comp_embeds,
        comp,
    ]
    components_att = [
        att_mask,
        TrainingConfig.attention_mask['P_Re_att'],
        pf_att,
        TrainingConfig.attention_mask['user_att'],
        ind_att,
        TrainingConfig.attention_mask['Tech_att'],
        tech_att,
        TrainingConfig.attention_mask['Comp_att'],
        comp_att,
    ]
    input_embeddings = torch.cat(components, dim=1)
    att_mask = torch.cat(components_att,dim=-1)

    # 手动释放缓存 
    torch.cuda.empty_cache()


    result_dict['input_embeding'] = input_embeddings
    result_dict['attention'] = att_mask
    return result_dict





def fuse4_DL(TrainingConfig,model,LLMs_token,fusion,projector,img_CF_feature,img_PF_feature,img_CL_feature,labels,att,embeds,ind_att,ind,tech_att,tech,comp_att,comp,pf_att,pf):

    
    result_dict = {}
    views = {}

    #扩展提示嵌入以匹配视觉特征
    expand_part1_embeds = TrainingConfig.prompt_embed['part1_embeds'].expand(TrainingConfig.batch_size,-1,-1).detach()
    expand_CF_embeds = TrainingConfig.prompt_embed['CF_embeds'].expand(TrainingConfig.batch_size,-1,-1).detach()
    expand_CFCL_embeds = TrainingConfig.prompt_embed['CFCL_embeds'].expand(TrainingConfig.batch_size,-1,-1).detach()
    expand_CFPF_embeds = TrainingConfig.prompt_embed['CFPF_embeds'].expand(TrainingConfig.batch_size,-1,-1).detach()
    expand_All_embeds = TrainingConfig.prompt_embed['All_embeds'].expand(TrainingConfig.batch_size,-1,-1).detach()
    expand_ind_embeds = TrainingConfig.prompt_embed['ind_embeds'].expand(TrainingConfig.batch_size,-1,-1).detach()
    expand_pf_embeds = TrainingConfig.prompt_embed['P_Re_embeds'].expand(TrainingConfig.batch_size,-1,-1).detach()
    expand_Tech_embeds = TrainingConfig.prompt_embed['Tech_embeds'].expand(TrainingConfig.batch_size,-1,-1).detach()
    expand_Comp_embeds = TrainingConfig.prompt_embed['Comp_embeds'].expand(TrainingConfig.batch_size,-1,-1).detach()
    expand_user_embeds = TrainingConfig.prompt_embed['user_embeds'].expand(TrainingConfig.batch_size,-1,-1).detach()

    # 确定输入类型
    if isinstance(img_PF_feature, torch.Tensor) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL + PF'
        views['cf'] = img_CF_feature    #[1,1369,768]
        views['cl'] = img_CL_feature    #[1,1369,768]
        views['pf'] = img_PF_feature    #[1,1369,768]
    elif isinstance(img_PF_feature, torch.Tensor) and (not isinstance(img_CL_feature, torch.Tensor)):
        input_type = 'CF + PF'
        views['cf'] = img_CF_feature    #[1,1369,768]
        views['pf'] = img_PF_feature    #[1,1369,768]
    elif (not isinstance(img_PF_feature, torch.Tensor)) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL'
        views['cf'] = img_CF_feature    #[1,1369,768]
        views['cl'] = img_CL_feature     #[1,1369,768]
    else:
        input_type = 'CF only'
        views['cf'] = img_CF_feature    #[1,1369,768]
    result_dict['input_type'] = input_type

    #注意力机制融合信息
    fusion_result = fusion(features_dict=views)
    
    #投影到文本空间
    vision_embeds = projector(fusion_result.to(next(projector.parameters()).device))
    vision_embeds = vision_embeds.to(model.device)
    if input_type == 'CF only':
        input_embeddings = torch.cat([expand_part1_embeds,expand_CF_embeds,vision_embeds],dim=1)
        att_mask = torch.cat([TrainingConfig.attention_mask['part1_att'], TrainingConfig.attention_mask['CF_att']], dim=-1)
        att_mask = F.pad(att_mask, (0,vision_embeds.shape[1]), value=1)

    elif input_type == 'CF + CL + PF':
        input_embeddings = torch.cat([expand_part1_embeds,expand_All_embeds,vision_embeds],dim=1)
        att_mask = torch.cat([TrainingConfig.attention_mask['part1_att'], TrainingConfig.attention_mask['All_att']], dim=-1)
        att_mask = F.pad(att_mask, (0,vision_embeds.shape[1]), value=1)

    elif input_type == 'CF + PF':
        input_embeddings = torch.cat([expand_part1_embeds,expand_CFPF_embeds,vision_embeds],dim=1)
        att_mask = torch.cat([TrainingConfig.attention_mask['part1_att'], TrainingConfig.attention_mask['CFPF_att']], dim=-1)
        att_mask = F.pad(att_mask, (0,vision_embeds.shape[1]), value=1)

    else:
        input_embeddings = torch.cat([expand_part1_embeds,expand_CFCL_embeds,vision_embeds],dim=1)
        att_mask = torch.cat([TrainingConfig.attention_mask['part1_att'], TrainingConfig.attention_mask['CFCL_att']], dim=-1)
        att_mask = F.pad(att_mask, (0,vision_embeds.shape[1]), value=1)
        
    # print(input_type)
    pf = pf.to(model.device).detach()
    ind = ind.to(model.device).detach()
    tech = tech.to(model.device).detach()
    comp = comp.to(model.device).detach()

    components = [
        input_embeddings,
        expand_ind_embeds,
        ind,
        expand_Tech_embeds,
        tech,
        expand_Comp_embeds,
        comp,
        expand_pf_embeds,
        pf,
        expand_user_embeds,
    ]
    # print(P_Re_att.shape,pf_att.shape,user_att.shape,ind_att.shape,Tech_att.shape,tech_att.shape,Comp_att.shape,comp_att.shape, att.shape)
    components_att = [
        att_mask,
        TrainingConfig.attention_mask['ind_att'],
        ind_att,
        TrainingConfig.attention_mask['Tech_att'],
        tech_att,
        TrainingConfig.attention_mask['Comp_att'],
        comp_att,
        TrainingConfig.attention_mask['P_Re_att'],
        pf_att,
        TrainingConfig.attention_mask['user_att'],
        att
    ]
    # print(f"注意力掩码形状为{att_mask.shape}")
    input_embeddings = torch.cat(components, dim=1)
    att_mask = torch.cat(components_att,dim=-1)
    # print(f"注意力掩码形状为{att_mask[:,1416:1500]}")

    #将labels进行改变
    # print(LLMs_token.pad_token_id)
    labels = labels.masked_fill(labels == LLMs_token.pad_token_id, -100)
    labels = F.pad(labels, (input_embeddings.shape[1],0), value=-100)
    labels = labels.to(model.device, dtype=torch.int64)
    #将目标文本进行嵌入
    input_embeddings = torch.cat((input_embeddings,embeds.to(model.device)),dim=1)
    embeds = embeds.detach()

    # 手动释放缓存 
    torch.cuda.empty_cache()


    result_dict['input_embeding'] = input_embeddings
    result_dict['attention'] = att_mask
    result_dict['labels'] = labels
    return result_dict


def fuse4_DL_test(TrainingConfig,model,fusion,projector,img_CF_feature,img_PF_feature,img_CL_feature,ind_att,ind,tech_att,tech,comp_att,comp,pf_att,pf):

    
    result_dict = {}
    views = {}

    #扩展提示嵌入以匹配视觉特征
    expand_part1_embeds = TrainingConfig.prompt_embed['part1_embeds'].expand(TrainingConfig.batch_size,-1,-1).detach()
    expand_CF_embeds = TrainingConfig.prompt_embed['CF_embeds'].expand(TrainingConfig.batch_size,-1,-1).detach()
    expand_CFCL_embeds = TrainingConfig.prompt_embed['CFCL_embeds'].expand(TrainingConfig.batch_size,-1,-1).detach()
    expand_CFPF_embeds = TrainingConfig.prompt_embed['CFPF_embeds'].expand(TrainingConfig.batch_size,-1,-1).detach()
    expand_All_embeds = TrainingConfig.prompt_embed['All_embeds'].expand(TrainingConfig.batch_size,-1,-1).detach()
    expand_ind_embeds = TrainingConfig.prompt_embed['ind_embeds'].expand(TrainingConfig.batch_size,-1,-1).detach()
    expand_pf_embeds = TrainingConfig.prompt_embed['P_Re_embeds'].expand(TrainingConfig.batch_size,-1,-1).detach()
    expand_Tech_embeds = TrainingConfig.prompt_embed['Tech_embeds'].expand(TrainingConfig.batch_size,-1,-1).detach()
    expand_Comp_embeds = TrainingConfig.prompt_embed['Comp_embeds'].expand(TrainingConfig.batch_size,-1,-1).detach()
    expand_user_embeds = TrainingConfig.prompt_embed['user_embeds'].expand(TrainingConfig.batch_size,-1,-1).detach()

    # 确定输入类型
    if isinstance(img_PF_feature, torch.Tensor) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL + PF'
        views['cf'] = img_CF_feature    #[1,1369,768]
        views['cl'] = img_CL_feature    #[1,1369,768]
        views['pf'] = img_PF_feature    #[1,1369,768]
    elif isinstance(img_PF_feature, torch.Tensor) and (not isinstance(img_CL_feature, torch.Tensor)):
        input_type = 'CF + PF'
        views['cf'] = img_CF_feature    #[1,1369,768]
        views['pf'] = img_PF_feature    #[1,1369,768]
    elif (not isinstance(img_PF_feature, torch.Tensor)) and isinstance(img_CL_feature, torch.Tensor):
        input_type = 'CF + CL'
        views['cf'] = img_CF_feature    #[1,1369,768]
        views['cl'] = img_CL_feature     #[1,1369,768]
    else:
        input_type = 'CF only'
        views['cf'] = img_CF_feature    #[1,1369,768]
    result_dict['input_type'] = input_type

    #注意力机制融合信息
    fusion_result = fusion(features_dict=views)
    
    #投影到文本空间
    vision_embeds = projector(fusion_result.to(next(projector.parameters()).device))
    vision_embeds = vision_embeds.to(model.device)
    if input_type == 'CF only':
        input_embeddings = torch.cat([expand_part1_embeds,expand_CF_embeds,vision_embeds],dim=1)
        att_mask = torch.cat([TrainingConfig.attention_mask['part1_att'], TrainingConfig.attention_mask['CF_att']], dim=-1)
        att_mask = F.pad(att_mask, (0,vision_embeds.shape[1]), value=1)

    elif input_type == 'CF + CL + PF':
        input_embeddings = torch.cat([expand_part1_embeds,expand_All_embeds,vision_embeds],dim=1)
        att_mask = torch.cat([TrainingConfig.attention_mask['part1_att'], TrainingConfig.attention_mask['All_att']], dim=-1)
        att_mask = F.pad(att_mask, (0,vision_embeds.shape[1]), value=1)

    elif input_type == 'CF + PF':
        input_embeddings = torch.cat([expand_part1_embeds,expand_CFPF_embeds,vision_embeds],dim=1)
        att_mask = torch.cat([TrainingConfig.attention_mask['part1_att'], TrainingConfig.attention_mask['CFPF_att']], dim=-1)
        att_mask = F.pad(att_mask, (0,vision_embeds.shape[1]), value=1)

    else:
        input_embeddings = torch.cat([expand_part1_embeds,expand_CFCL_embeds,vision_embeds],dim=1)
        att_mask = torch.cat([TrainingConfig.attention_mask['part1_att'], TrainingConfig.attention_mask['CFCL_att']], dim=-1)
        att_mask = F.pad(att_mask, (0,vision_embeds.shape[1]), value=1)
        
    # print(input_type)
    pf = pf.to(model.device).detach()
    ind = ind.to(model.device).detach()
    tech = tech.to(model.device).detach()
    comp = comp.to(model.device).detach()

    components = [
        input_embeddings,
        expand_ind_embeds,
        ind,
        expand_Tech_embeds,
        tech,
        expand_Comp_embeds,
        comp,
        expand_pf_embeds,
        pf,
        expand_user_embeds,
    ]
    # print(P_Re_att.shape,pf_att.shape,user_att.shape,ind_att.shape,Tech_att.shape,tech_att.shape,Comp_att.shape,comp_att.shape, att.shape)
    components_att = [
        att_mask,
        TrainingConfig.attention_mask['ind_att'],
        ind_att,
        TrainingConfig.attention_mask['Tech_att'],
        tech_att,
        TrainingConfig.attention_mask['Comp_att'],
        comp_att,
        TrainingConfig.attention_mask['P_Re_att'],
        pf_att,
        TrainingConfig.attention_mask['user_att'],
    ]
    # print(f"注意力掩码形状为{att_mask.shape}")
    input_embeddings = torch.cat(components, dim=1)
    att_mask = torch.cat(components_att,dim=-1)
    # print(f"注意力掩码形状为{att_mask[:,1416:1500]}")


    # 手动释放缓存 
    torch.cuda.empty_cache()


    result_dict['input_embeding'] = input_embeddings
    result_dict['attention'] = att_mask
    # result_dict['labels'] = labels
    return result_dict


def fuse4_batches(TrainingConfig,model,LLMs_token,fusion,projector,img_CF_feature,img_PF_feature,img_CL_feature,labels,att,embeds,ind_att,ind,tech_att,tech,comp_att,comp,pf_att,pf,valid_masks):
    result_dict = {}
    views = {}

    # 1. 获取所有系统提示嵌入
    sys_prompts = {
        'cf_only': TrainingConfig.prompt_embed['sys_embeds'],    # [1,30,4096]
        'cf_cl': TrainingConfig.prompt_embed['sys3_embeds'],     # [1,41,4096]
        'cf_pf': TrainingConfig.prompt_embed['sys2_embeds'],     # [1,35,4096] 示例
        'cf_cl_pf': TrainingConfig.prompt_embed['sys1_embeds']   # [1,45,4096] 示例
    }
    
    # 2. 计算最大序列长度
    max_seq_len = max(p.shape[1] for p in sys_prompts.values())  # 假设为45

    # 3. 统一填充所有提示到相同长度
    padded_sys_embeds = []
    for key in ['cf_only', 'cf_cl', 'cf_pf', 'cf_cl_pf']:
        prompt = sys_prompts[key]
        pad_size = max_seq_len - prompt.shape[1]
        
        # 使用零填充左侧
        padded_prompt = F.pad(prompt, (0, 0, pad_size,0))  # (D维度, 序列维度)
        padded_sys_embeds.append(padded_prompt)

    # 4. 创建掩码标识有效部分
    sys_attention_masks = {
        'cf_only': F.pad(TrainingConfig.attention_mask['sys_att'], (0,max_seq_len-TrainingConfig.attention_mask['sys_att'].shape[1])).to(torch.float32),
        'cf_cl': F.pad(TrainingConfig.attention_mask['sys3_att'], (0,max_seq_len-TrainingConfig.attention_mask['sys3_att'].shape[1])).to(torch.float32),
        'cf_pf': F.pad(TrainingConfig.attention_mask['sys2_att'], (0,max_seq_len-TrainingConfig.attention_mask['sys2_att'].shape[1])).to(torch.float32),
        'cf_cl_pf': F.pad(TrainingConfig.attention_mask['sys1_att'], (0,max_seq_len-TrainingConfig.attention_mask['sys1_att'].shape[1])).to(torch.float32),
    }

    has_pf = valid_masks[:, 1].bool()  # [B]
    has_cl = valid_masks[:, 2].bool()  # [B]

    # 创建类型编码矩阵 [B,4] 对应四种类型
    type_matrix = torch.stack([
        (~has_pf) & (~has_cl),  # CF only
        (~has_pf) & has_cl,     # CF + CL
        has_pf & (~has_cl),     # CF + PF
        has_pf & has_cl         # CF + CL + PF
    ], dim=1).float()  # [B,4]

    # ==== 2. 动态选择系统提示 ====
    # 预加载所有系统提示嵌入 [4, L, D]
    sys_embeds = torch.stack(padded_sys_embeds)

    # 为每个样本选择对应的系统提示 [B, L, D]
    sys_indices = type_matrix.argmax(dim=1)  # [B]
    selected_sys_embeds = sys_embeds[sys_indices].squeeze(1)  # [B, L, D]
    true_max_seq_len = sys_embeds.size(2)
    selected_sys_att = torch.zeros(TrainingConfig.batch_size, true_max_seq_len, device=sys_embeds.device, dtype=torch.float32)
    # 为每个类型填充掩码
    for i, key in enumerate(['cf_only', 'cf_cl', 'cf_pf', 'cf_cl_pf']):
        mask = (sys_indices == i)
        num_samples = mask.sum().item()
        if num_samples > 0:
            # 扩展原始掩码到匹配样本数 [num_samples, max_seq_len]
            expanded_mask = sys_attention_masks[key].expand(num_samples, -1).to(
            device=selected_sys_att.device,
            dtype=selected_sys_att.dtype  # 显式转换类型
        )
            selected_sys_att[mask] = expanded_mask

    #扩展提示嵌入以匹配视觉特征
    expand_user_embeds = TrainingConfig.prompt_embed['user_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_usual1_embeds = TrainingConfig.prompt_embed['usual1_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_pf_embeds = TrainingConfig.prompt_embed['P_Re_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Tech_embeds = TrainingConfig.prompt_embed['Tech_embeds'].expand(TrainingConfig.batch_size,-1,-1)
    expand_Comp_embeds = TrainingConfig.prompt_embed['Comp_embeds'].expand(TrainingConfig.batch_size,-1,-1)


    views['cf'] = img_CF_feature    #[B,1369,768]
    views['cl'] = img_CL_feature    #[B,1369,768]
    views['pf'] = img_PF_feature    #[B,1369,768]


    #注意力机制融合信息
    fusion_result = fusion(features_dict=views,valid_masks=valid_masks)
    
    #投影到文本空间
    vision_embeds = projector(fusion_result.to(next(projector.parameters()).device))
    vision_embeds = vision_embeds.to(model.device)

    # 拼接核心组件 [B, L_sys + L_vis + L_usual, D]
    main_components = torch.cat([
        selected_sys_embeds.to(vision_embeds.device),
        vision_embeds,
        expand_usual1_embeds.to(vision_embeds.device)
    ], dim=1)


    # ==== 6. 注意力掩码生成 ====


     # 动态生成掩码 [B, L_total]
    sys_att = selected_sys_att.to(vision_embeds.device)
    vis_att = torch.ones(TrainingConfig.batch_size, vision_embeds.size(1))
    usual_att = TrainingConfig.attention_mask['usual1_att'].expand(TrainingConfig.batch_size, -1)
    main_att = torch.cat([sys_att, vis_att, usual_att], dim=1)


    # ==== 7. 附加组件处理 ====
    # 其他固定组件（pf、tech等）
    pf = pf.to(model.device).detach()
    ind = ind.to(model.device).detach()
    tech = tech.to(model.device).detach()
    comp = comp.to(model.device).detach()
    embeds = embeds.to(model.device).detach()
    extra_components = [
        main_components,
        expand_pf_embeds,
        pf,
        expand_user_embeds,
        ind,
        expand_Tech_embeds,
        tech,
        expand_Comp_embeds,
        comp,
        embeds,
    ]
    
    # 拼接所有组件
    input_embeddings = torch.cat(extra_components, dim=1)
    
    # ==== 8. 注意力掩码拼接 ====
    extra_att = [
        main_att,
        TrainingConfig.attention_mask['P_Re_att'].expand(TrainingConfig.batch_size, -1),
        pf_att,
        TrainingConfig.attention_mask['user_att'].expand(TrainingConfig.batch_size, -1),
        ind_att,
        TrainingConfig.attention_mask['Tech_att'].expand(TrainingConfig.batch_size, -1),
        tech_att,
        TrainingConfig.attention_mask['Comp_att'].expand(TrainingConfig.batch_size, -1),
        comp_att,
        att
    ]
    att_mask = torch.cat(extra_att, dim=1)


    #将labels进行改变
    # print(LLMs_token.pad_token_id)
    labels = labels.masked_fill(labels == LLMs_token.pad_token_id, -100)
    labels = F.pad(labels, (input_embeddings.shape[1],0), value=-100)
    labels = labels.to(model.device, dtype=torch.int64)
    

    # 手动释放缓存 
    torch.cuda.empty_cache()


    result_dict['input_embeding'] = input_embeddings
    result_dict['attention'] = att_mask
    result_dict['labels'] = labels
    return result_dict

