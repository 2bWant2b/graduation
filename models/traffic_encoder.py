import torch
import torch.nn as nn
import timm


class TrafficEncoder(nn.Module):
    def __init__(
            self,
            num_classes: int = 0,  # 默认为0，因为我们只用它提取特征
            model_name: str = 'vit_base_patch16_224',
            use_prompt: bool = True,
            prompt_length: int = 10,  # 默认 Prompt 长度
            pretrained: bool = True
    ):
        super(TrafficEncoder, self).__init__()

        # 1. 加载 ViT 底座 (使用 timm)
        print(f"Loading ViT Backbone: {model_name}...")
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

        # 获取嵌入维度 (ViT-Base 通常是 768)
        self.embed_dim = self.backbone.embed_dim

        # 2. 定义可学习的 Prompt (如果需要独立使用)
        self.use_prompt = use_prompt
        self.prompt_length = prompt_length
        if use_prompt:
            # 定义一个默认的 Prompt 向量 [1, Prompt_Len, Embed_Dim]
            self.prompt_vector = nn.Parameter(torch.randn(1, prompt_length, self.embed_dim))
            nn.init.normal_(self.prompt_vector, std=0.02)
            self.prompt_dropout = nn.Dropout(0.1)

    def forward(self, x, return_features=True):
        """
        普通前向传播
        x: [Batch, 3, 224, 224]
        """
        # 提取 Patch Embeddings
        x = self.backbone.patch_embed(x)

        # 加上 Class Token 和 Positional Embeddings
        # 注意：这里需要手动处理 timm 的内部逻辑，或者使用 forward_features
        # 为了简单，我们直接利用 timm 的 forward_features，但在中间截断插入 Prompt 比较麻烦
        # 所以这里我们采用 "Late Fusion" 或 "Input Concatenation" 策略

        # 简单实现：使用 backbone 提取特征
        features = self.backbone.forward_features(x)  # [B, N_patches+1, 768]

        # 取 [CLS] token
        cls_token = features[:, 0, :]

        if return_features:
            return cls_token
        return cls_token  # 如果有分类头，这里接分类头

    def forward_instruct(
            self,
            vision_input,
            prompt_embeddings=None,  # <--- 这里的 Prompt 由日志生成
            return_features=True
    ):
        """
        核心方法：支持 MoPE 的指令注入
        vision_input: [Batch, 3, 224, 224]
        prompt_embeddings: [Batch, Prompt_Len, 768] (来自 Log Encoder)
        """
        # 1. 获取 ViT 的 Patch Embeddings [B, N_patches, 768]
        x = self.backbone.patch_embed(vision_input)

        # 2. 拼接 CLS Token (如果 backbone 里还没加)
        cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # 3. 加上位置编码
        x = self.backbone.pos_drop(x + self.backbone.pos_embed)

        # 4. === 关键：插入日志生成的 Prompt ===
        if prompt_embeddings is not None:
            # prompt_embeddings shape: [B, Prompt_Len, 768]
            # 把它拼接在序列的前面 (在 CLS 之后，或者之前)
            # 这里选择拼接在 CLS 之后
            x = torch.cat([
                x[:, :1, :],  # CLS
                prompt_embeddings,  # Log Prompt
                x[:, 1:, :]  # Image Patches
            ], dim=1)

        # 5. 通过 Transformer Block
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)

        # 6. 取出 CLS Token 作为最终特征
        cls_out = x[:, 0, :]

        if return_features:
            # 为了兼容 mope.py 的接口，返回 (logits, features)
            # 这里 logits 设为 None
            return None, cls_out
        return cls_out