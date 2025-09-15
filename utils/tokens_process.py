import math
from typing import List, Tuple
from PIL import Image, ImageDraw
import torch.nn.functional as F
import torch
import torch.nn as nn
import random
import numpy as np

class ViewTypeEncoder(nn.Module):
    def __init__(self, num_view_types: int, embed_dim: int):
        """
        初始化视图类型编码器。
        参数:
            num_view_types (int): 视图类型的总数。
            embed_dim (int): 特征的维度，类型编码将添加到这个维度上。
        """
        super().__init__()
        self.type_embedding = nn.Embedding(num_view_types, embed_dim)
        
        # 为了方便，可以定义一些常量来表示类型ID
        # 这些ID应该从0开始，连续到 num_view_types - 1
        self.VIEW_CF_PATCH_ID = 0
        self.VIEW_CL_PATCH_ID = 1
        self.VIEW_PF_PATCH_ID = 2
        self.VIEW_CF_GLOBAL_ID = 3
        self.VIEW_CL_GLOBAL_ID = 4
        self.VIEW_PF_GLOBAL_ID = 5
        self.VIEW_FUSED_GLOBAL_ID = 6
        # self.VIEW_DELTA_ID = 4   # <--- 为差异流视图（V_delta）添加新ID
        # ... 根据您的需要添加更多类型

    def forward(self, features: torch.Tensor, view_type_id: int) -> torch.Tensor:
        """
        将类型编码添加到输入特征中。
        参数:
            features (torch.Tensor): 输入特征，形状通常为 [B, N, D] (批次, 序列长度, 维度)
                                     或 [N, D] (如果批次大小为1且被压缩)
            view_type_id (int): 当前特征的视图类型ID。
        返回:
            torch.Tensor: 添加了类型编码的特征。
        """
        B, N, D = features.shape if features.ndim == 3 else (1, features.shape[0], features.shape[1])
        if features.ndim == 2: # 如果是 [N, D]，先扩展为 [1, N, D]
            features = features.unsqueeze(0)

        # 获取类型编码向量，形状为 [1, D]
        type_emb = self.type_embedding(torch.tensor([view_type_id], device=features.device))
        
        # 将类型编码扩展并添加到特征中
        # type_emb 形状 [1, D] -> [1, 1, D] 以便与 [B, N, D] 的特征相加
        features_with_type = features + type_emb.unsqueeze(1) 
        
        if features.ndim == 2: # 如果原始是 [N,D], 恢复形状
             return features_with_type.squeeze(0)
        return features_with_type\
        
class ViewTypeEncoder_3(nn.Module):
    def __init__(self, num_view_types: int, embed_dim: int):
        """
        初始化视图类型编码器。
        参数:
            num_view_types (int): 视图类型的总数。
            embed_dim (int): 特征的维度，类型编码将添加到这个维度上。
        """
        super().__init__()
        self.type_embedding = nn.Embedding(num_view_types, embed_dim)
        
        # 为了方便，可以定义一些常量来表示类型ID
        # 这些ID应该从0开始，连续到 num_view_types - 1
        self.VIEW_CF_GLOBAL_ID = 0
        self.VIEW_CL_GLOBAL_ID = 1
        self.VIEW_PF_GLOBAL_ID = 2
        self.VIEW_FUSED_GLOBAL_ID = 3
        self.VIEW_DELTA_ID = 4   # <--- 为差异流视图（V_delta）添加新ID
        # ... 根据您的需要添加更多类型

    def forward(self, features: torch.Tensor, view_type_id: int) -> torch.Tensor:
        """
        将类型编码添加到输入特征中。
        参数:
            features (torch.Tensor): 输入特征，形状通常为 [B, N, D] (批次, 序列长度, 维度)
                                     或 [N, D] (如果批次大小为1且被压缩)
            view_type_id (int): 当前特征的视图类型ID。
        返回:
            torch.Tensor: 添加了类型编码的特征。
        """
        B, N, D = features.shape if features.ndim == 3 else (1, features.shape[0], features.shape[1])
        if features.ndim == 2: # 如果是 [N, D]，先扩展为 [1, N, D]
            features = features.unsqueeze(0)

        # 获取类型编码向量，形状为 [1, D]
        type_emb = self.type_embedding(torch.tensor([view_type_id], device=features.device))
        
        # 将类型编码扩展并添加到特征中
        # type_emb 形状 [1, D] -> [1, 1, D] 以便与 [B, N, D] 的特征相加
        features_with_type = features + type_emb.unsqueeze(1) 
        
        if features.ndim == 2: # 如果原始是 [N,D], 恢复形状
             return features_with_type.squeeze(0)
        return features_with_type

def choose_best_grid(
    image_height: int,
    image_width: int,
    target_size: int = 518,
    # candidate_grids: List[Tuple[int, int]] = [(1, 1), (1, 2), (2, 1), (2, 2), (1,3), (3,1), (2, 3), (3, 2), (3, 3)]
    # candidate_grids: List[Tuple[int, int]] = [(1, 1), (1, 2), (2, 1), (2, 2), (1,3), (3,1), (2, 3), (3, 2)]
    candidate_grids: List[Tuple[int, int]] = [(1, 1)]
) -> Tuple[int, int]:
    """
    Choose the best grid configuration for splitting the image based on minimal resizing error
    and accounting for medical image characteristics (sparse useful information).
    """
    best_grid = (1, 1)
    min_error = float('inf')

    for h_grid, w_grid in candidate_grids:
        patch_height = image_height / h_grid
        patch_width = image_width / w_grid

        # Penalize too small patches (below 400x400), they tend to be less informative
        if patch_height < 400 or patch_width < 400:
            continue

        # Penalize too many patches (which can dilute important signals)
        num_patches = h_grid * w_grid
        sparsity_penalty = num_patches * 0.05  # adjustable weight

        # Distance from ideal patch size
        error = math.sqrt((patch_height - target_size) ** 2 + (patch_width - target_size) ** 2)
        total_score = error + sparsity_penalty

        if total_score < min_error:
            min_error = total_score
            best_grid = (h_grid, w_grid)

    return best_grid

def build_2d_sinusoidal_pos_embed(H, W, D, device):
    """
    构造 patch 内部 2D 位置编码（H=高, W=宽, D=维度）
    返回 shape: [H*W, D]
    """
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid_y = grid_y.reshape(-1).float().to(device)  # [H*W]
    grid_x = grid_x.reshape(-1).float().to(device)

    dim_t = torch.arange(D // 2, dtype=torch.float32, device=device)
    dim_t = 10000 ** (2 * (dim_t // 2) / (D // 2))

    # 分别编码 x 和 y
    pos_x = grid_x[:, None] / dim_t  # [H*W, D/2]
    pos_y = grid_y[:, None] / dim_t  # [H*W, D/2]

    # 拼接
    pos_x_embed = torch.cat([torch.sin(pos_x), torch.cos(pos_x)], dim=-1)  # [H*W, D]
    pos_y_embed = torch.cat([torch.sin(pos_y), torch.cos(pos_y)], dim=-1)  # [H*W, D]

    # 合并为最终编码
    pos_embed = torch.cat([pos_y_embed[:, :D // 2], pos_x_embed[:, :D // 2]], dim=-1)  # [H*W, D]
    return pos_embed  # [H*W, D]

def build_patch_location_encoding(M, N, D, device):
    """
    生成 patch 级别的全局位置编码，用于描述每个 patch 在图像网格中的相对位置。
    
    参数:
        M: 图像在垂直方向被切成的 patch 数（行数）
        N: 图像在水平方向被切成的 patch 数（列数）
        D: 输出位置编码的维度（必须是偶数）
        device: 放置生成编码的设备

    返回:
        Tensor of shape [M*N, D]：每个 patch 的位置编码
    """
    assert D % 2 == 0, "Embedding dimension D must be even."

    # 创建网格坐标
    grid_y, grid_x = torch.meshgrid(torch.arange(M), torch.arange(N), indexing='ij')  # [M, N]
    pos = torch.stack([grid_y, grid_x], dim=-1).reshape(-1, 2).float().to(device)  # [M*N, 2]

    # 准备维度展开比例（对 D/2 进行编码）
    dim_t = torch.arange(D // 2, dtype=torch.float32, device=device)
    dim_t = 10000 ** (2 * (dim_t // 2) / (D // 2))

    # 对 x 和 y 坐标分别编码为 D/2 维
    pos_x = pos[:, 1:2] / dim_t  # [M*N, D/2]
    pos_y = pos[:, 0:1] / dim_t  # [M*N, D/2]

    # 拼接 sin/cos 形式的位置编码
    pos_x_embed = torch.cat([torch.sin(pos_x), torch.cos(pos_x)], dim=-1)  # [M*N, D]
    pos_y_embed = torch.cat([torch.sin(pos_y), torch.cos(pos_y)], dim=-1)  # [M*N, D]

    # 合并为最终位置编码，每个方向一半
    pos_embed = torch.cat([pos_y_embed[:, :D // 2], pos_x_embed[:, :D // 2]], dim=-1)  # [M*N, D]

    return pos_embed  # [B, D]

def add_dual_positional_encoding(patches_feature, M, N, device):
    """
    patches_feature: [B, 1369, 768]
    M, N: patch 图像在原图中分割的网格数
    输出: 加入位置编码的特征 [B, 1369, 768]
    """
    B, T, D = patches_feature.shape  # T=1369, D=768

    # Patch 内 token 编码
    grid_dim = int(math.sqrt(T))
    assert grid_dim * grid_dim == T, f"Token数量必须为平方数，当前为 {T}"
    inner_pos_embed = build_2d_sinusoidal_pos_embed(grid_dim, grid_dim, D, device)  # [T, D]

    # Patch 位置编码
    outer_pos_embed = build_patch_location_encoding(M, N, D, device)  # [B, D]

    # 添加编码
    patches_feature = patches_feature + inner_pos_embed.unsqueeze(0)  # broadcast T # [B, 1369, 768] + [1, 1369, 768]
    patches_feature = patches_feature + outer_pos_embed.unsqueeze(1)  # broadcast B     [B, 1369, 768] + [B, 1, 768]

    return patches_feature


def ResAny(image,visualize = False):
    W, H = image.size
    M, N = choose_best_grid(W,H)
    print(M,N)
    patch_w, patch_h = W // N, H // M
    print(f"此图像的大小为{W}*{H}")
    patches = []
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)
    for i in range(M):
        for j in range(N):
            left = j * patch_w
            top = i * patch_h

            right = (j + 1) * patch_w if j < N - 1 else W
            bottom = (i + 1) * patch_h if i < M - 1 else H

            crop = image.crop((left, top, right, bottom))
            patches.append(crop)
            if visualize:
                draw.rectangle([left, top, right, bottom], outline="red", width=3)

    # if len(patches) <= max_patches:
    #     return patches,M,N
    
    # return random.sample(patches, k=max_patches)
    return patches,M,N
    # return patches, M,N, vis_image if visualize else None

def resize_tokens_interpolate(vision_embeds, target_len):
    vision_embeds = vision_embeds.permute(0, 2, 1)  # [B, D, L]
    vision_embeds = F.interpolate(vision_embeds, size=target_len, mode='linear', align_corners=False)
    vision_embeds = vision_embeds.permute(0, 2, 1)  # [B, target_len, D]
    return vision_embeds
    
class VisualTokenCompressor(nn.Module):
    """
    用于从原始视觉 token 中压缩出更少的 summary token。
    例如：从 1370 token -> 440 token
    """
    def __init__(self, input_dim: int = 768, output_tokens: int = 440, num_heads: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.output_tokens = output_tokens

        # 使用轻量 transformer encoder 对视觉 token 聚合
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8, dim_feedforward=2048, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.ln = nn.LayerNorm(input_dim)

        # Learnable Query token，用于 cross-attn 提取摘要
        self.query_tokens = nn.Parameter(torch.empty(1, output_tokens, input_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        self.cross_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)

        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Linear(input_dim * 4, input_dim),
        )

        # Token importance estimator,高分关注
        # self.importance_proj = nn.Sequential(
        #     nn.Linear(input_dim, 1),
        #     nn.Sigmoid()
        # )

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        输入:
            visual_tokens: Tensor, shape [B, N, D]
        输出:
            summary_tokens: Tensor, shape [B, output_tokens, D]
        """
        B, N, D = visual_tokens.shape

        # Step 1: self-attention encoder 聚合 token 表达
        encoded = self.encoder(visual_tokens)  # [B, N, D]
        encoded = self.ln(encoded)

        # Estimate token importance
        # importance = self.importance_proj(encoded)  # [B, N, 1]
        # weighted_encoded = encoded * importance  # [B, N, D]

        # Step 2: cross-attn，从 learnable query 中提取关键表示
        query = self.query_tokens.expand(B, -1, -1)  # [B, output_tokens, D]
        summary, _ = self.cross_attn(query, encoded, encoded)  # [B, output_tokens, D]
        # summary, _ = self.cross_attn(query, weighted_encoded, weighted_encoded)  # [B, output_tokens, D]

        summary = summary + self.mlp(summary)  # residual

        return summary

class VisualTokenCompressorV2(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        output_tokens: int = 128,
        num_query_groups: int = 2,        # 视图数量，如 CF、CL、PF
        layers: int = 4,
        num_heads: int = 8,
        use_importance: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_tokens = output_tokens
        self.use_importance = use_importance
        self.num_query_groups = num_query_groups

        # 每组 learnable query
        self.query_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(1, output_tokens, input_dim)) for _ in range(num_query_groups)
        ])

        # token importance 估计器
        if self.use_importance:
            self.importance_proj = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, 1),
                nn.Sigmoid()
            )

        # 多层交替：CrossAttention + SelfAttention
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True),
                "cross_ln": nn.LayerNorm(input_dim),
                "self_attn": nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads,
                                                        dim_feedforward=2048, dropout=dropout, batch_first=True),
                "mlp": nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, input_dim * 4),
                    nn.GELU(),
                    nn.Linear(input_dim * 4, input_dim),
                )
            }))

        self.final_ln = nn.LayerNorm(input_dim)
    
    def forward(self, visual_tokens: torch.Tensor, view_id: int = 0):
        """
        输入：
            visual_tokens: [B, N, D]
            view_id: 用于选择哪一组 learnable query（如 CF/CL/PF）
            pos_embed: [B, N, D] or None，用于位置编码

        输出：
            summary_tokens: [B, output_tokens, D]
        """
        B, N, D = visual_tokens.shape

        if self.use_importance:
            importance = self.importance_proj(visual_tokens)   # [B, N, 1]
            visual_tokens = visual_tokens * (0.5 + 0.5 * importance)

        # 获取当前视图的 query token
        print(f'当前的视觉提取器使用的代号是{view_id}')
        query = self.query_tokens[view_id].expand(B, -1, -1)  # [B, T, D]

        for layer in self.layers:
            # --- Cross Attention ---
            query_norm = layer["cross_ln"](query)
            query, _ = layer["cross_attn"](query_norm, visual_tokens, visual_tokens)  # [B, T, D]

            # --- MLP (residual) ---
            query = query + layer["mlp"](query)

            # --- Self-Attention on queries ---
            query = layer["self_attn"](query)

        return self.final_ln(query)  # [B, T, D]


class VariableSummaryAttention(nn.Module):
    def __init__(self, 
                 input_dim: int = 768,
                 max_summary_tokens: int = 512,
                 num_heads: int = 8):
        """
        Args:
            input_dim: Dimension of visual tokens (e.g., 768 for ViT-base)
            max_summary_tokens: Maximum number of learnable summary tokens
            num_heads: Number of attention heads
        """
        super().__init__()
        self.input_dim = input_dim
        self.max_summary_tokens = max_summary_tokens

        # Learnable summary tokens
        self.summary_tokens = nn.Parameter(
            torch.randn(max_summary_tokens, input_dim)
        )

        # Cross-attention layer: summary tokens attend to visual tokens
        self.cross_attn = nn.MultiheadAttention(embed_dim=input_dim, 
                                                num_heads=num_heads, 
                                                batch_first=True)

        self.ln = nn.LayerNorm(input_dim)

    def forward(self, visual_tokens: torch.Tensor, target_len: int):
        """
        Args:
            visual_tokens: [B, N, D] - original visual tokens
            target_len: target summary token length (must <= max_summary_tokens)

        Returns:
            compressed: [B, target_len, D]
        """
        B, N, D = visual_tokens.shape
        assert target_len <= self.max_summary_tokens, \
            f"target_len={target_len} exceeds max_summary_tokens={self.max_summary_tokens}"

        # Select the first target_len summary tokens, expand to batch
        summary = self.summary_tokens[:target_len].unsqueeze(0).expand(B, -1, -1)  # [B, target_len, D]

        # Cross-attention: summary tokens attend to visual tokens
        # Query: summary tokens | Key, Value: visual tokens
        compressed, _ = self.cross_attn(query=summary, 
                                        key=visual_tokens,
                                        value=visual_tokens)

        compressed = self.ln(compressed)

        return compressed


# from PIL import Image, ImageDraw
# image_paths = ["./datasets/MIMIC-complete/mimic-cxr-images/files/p10/p10032409/s50002405/01afaa67-31d0e7ae-5e94ee5e-3f4d9deb-8b2be56f.jpg","./datasets/MIMIC-complete/mimic-cxr-images/files/p12/p12000091/s55199984/84f0ad73-7bf47610-aaef953b-1cee947a-39b63176.jpg",
# "./datasets/MIMIC-complete/mimic-cxr-images/files/p12/p12000091/s55199984/528b3a72-87c4c173-121695d8-8c8ab04a-a617dfdb.jpg","./datasets/MIMIC-complete/mimic-cxr-images/files/p12/p12000091/s58487107/94c5f631-e1e61da6-d972f176-45999116-5a34af51.jpg",
# "./datasets/MIMIC-complete/mimic-cxr-images/files/p12/p12000894/s50984094/1a4e5daf-5c6365b1-90567e79-9036268f-09254d78.jpg","./datasets/MIMIC-complete/mimic-cxr-images/files/p16/p16000035/s52654671/8e338050-c72628f4-cf19ef85-cb13d287-5af57beb.jpg",
# "./datasets/MIMIC-complete/mimic-cxr-images/files/p12/p12001854/s51877491/7636760d-998f01e0-a0054606-56afd977-af4967b5.jpg","./datasets/MIMIC-complete/mimic-cxr-images/files/p16/p16000035/s52654671/8e338050-c72628f4-cf19ef85-cb13d287-5af57beb.jpg",
# "./datasets/MIMIC-complete/mimic-cxr-images/files/p16/p16001249/s54426570/01954b31-905a80f8-1298cc54-7e47ae34-fdf291f2.jpg","datasets/MIMIC-complete/mimic-cxr-images/files/p16/p16002373/s57173558/685c08c4-1778cafb-85fc0228-1269ff9f-b6e62945.jpg",
# "./datasets/MIMIC-complete/mimic-cxr-images/files/p16/p16003901/s51119949/3c31d6bd-7489ef4c-5ea563c7-8b73be12-8d610a8e.jpg","./datasets/MIMIC-complete/mimic-cxr-images/files/p16/p16003901/s51119949/cdbfac98-544c58c7-bbe177be-b35ca6a2-bbc526ac.jpg",
# ]

# image_path =  image_paths[9] # 替换为你自己的图像路径
# image = Image.open(image_path).convert("RGB")

# patches, vis = ResAny(image, visualize=True)

# # 展示可视化图像
# if vis:
#     plt.figure(figsize=(8, 8))
#     plt.imshow(vis)
#     plt.axis("off")
#     plt.title("ResAny Patch Boundaries")
#     plt.show()



# pipe_image = pipeline(task="image-feature-extraction", model=Mcfg.dino_path, pool=False, device=0)
# rad_dino = AutoModel.from_pretrained(Mcfg.dino_path)
# processor = AutoImageProcessor.from_pretrained(Mcfg.dino_path)
# inputs = processor(images=image, return_tensors="pt")
# with torch.inference_mode():
#     outputs = rad_dino(**inputs)
#     features = outputs.last_hidden_state.squeeze(0)
# image_feature = torch.tensor(pipe_image(image_path)).squeeze(0)[1:,:]
# feature = features[1:,:]
# # print(torch.Tensor(image_feature).device)
# # print(torch.Tensor(features).device)
# # print(torch.allclose(image_feature, feature, atol=1e-4))  # True 则几乎一致
# print(type(feature))
