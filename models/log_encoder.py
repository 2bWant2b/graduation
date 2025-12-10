from transformers.models.bert.modeling_bert import (
    BertModel,
)
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast
from utils import debug_utils



class BertClassifier(nn.Module):
    """
    Bert for sequence classification, with prompt tuning support.
    """

    def __init__(
        self,
        num_classes: int,                                                           # 分类数量
        use_prompt: bool = False,                                                   # 是否开启Prompt Tuning（提示微调）
        prompt_length: int = 0,                                                     # Prompt向量的长度
        prompt_depth: str = "all",                                                  # Prompt插入深度，“all”表示每一层，“input”表示输入层
        use_instruct_moe=False,                                                     # 是否使用指令式MoE
        is_main_modal=False,                                                        # TODO 是否为主模态（影响专家数量和投影层初始化？）
    ) -> None:
        super(BertClassifier, self).__init__()
        self.bert_encoder = BertModel.from_pretrained("bert-base-uncased")          # 12层、768维的小写英文模型
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")     # TODO 分词器（什么意思？）
        self.num_classes = num_classes                                              # 分类数量
        self.heads = nn.Linear(self.bert_encoder.config.hidden_size, num_classes)   # 768->num_classes 分类头

        self.use_prompt = use_prompt
        self.prompt_length = prompt_length if use_prompt else 0
        # if use_instruct_moe:
        #     self.prompt_length = self.prompt_length * 2 + 1
        self.prompt_vector = None                                                   # 可学习的Prompt向量

        if use_prompt:
            if prompt_depth == "all":
                self.prompt_depth = self.bert_encoder.config.num_hidden_layers
            elif prompt_depth == "input":
                self.prompt_depth = 1
            self.prompt_depth = self.bert_encoder.config.num_hidden_layers          # 这里强制了Prompt深度为12，即BERT隐藏层数
            self.prompt_vector = nn.Parameter(                                      # Prompt形状：[Prompt深度(12), Prompt长度, 特征维度(768)]，即每一层都会插入这一层专属的Prompt
                torch.randn(
                    self.prompt_depth,
                    prompt_length,
                    self.bert_encoder.config.hidden_size,
                )
            )
            self.prompt_dropout = nn.Dropout(0.1)                                   # TODO Dropout防止过拟合
            nn.init.uniform_(self.prompt_vector, -0.3, 0.3)                      # 初始化Prompt参数
            self.freeze_backbone()                                                  # 冻结骨干，只训练self.prompt_vector和self.heads

        if True:  # this indentation for t2v, the normal pbert                      # 启用MoPE的能力

            # additional projection from vision
            self.prompt_proj_act = nn.GELU()                                        # 激活函数用于处理从视觉特征映射过来的向量
            self.prompt_vectors = []                                                # 专家列表
            #!HARDCODED Oct 13: assume vis model dim 768
            if is_main_modal:                                                       # TODO Mapping Layers初始化
                self.prompt_proj_0 = nn.Linear(384, 768)         # 根据ViT的输出维度来更改这里的384维度的输入
                self.prompt_proj_1 = nn.Linear(384, 768)
                self.prompt_proj_2 = nn.Linear(384, 768)

            for _ in range(self.bert_encoder.config.num_hidden_layers):             # 遍历BERT的每一层
                moe_n_experts = 16  if is_main_modal else 1                         # 如果是主模态则有16个动态专家
                _n_prompt_vec = moe_n_experts + 1                                   # 总Prompt数量 = 专家数 + 1个静态Prompt
                _prompt_vec = nn.Parameter(                                         # 定义这一层的专家参数，形状：[17, Prompt长度, 768]，含义：这一层有17个可选的Prompt，由Router决定用哪几个
                    torch.randn(_n_prompt_vec, prompt_length, 768)
                )
                nn.init.uniform_(_prompt_vec, -0.3, 0.3)                         # 初始化专家参数
                self.prompt_vectors.append(_prompt_vec)                             # 加入专家列表
            self.prompt_vectors = nn.ParameterList(self.prompt_vectors)
        # if True: # this indentation for moe v2t ,


    def forward(
        self,
        text_input,                         # TODO 输入的日志文本列表，如 ["POST /login...", "GET /index..."]，是代表一条日志吗？
        return_features=False,              # 是否返回[CLS]特征向量（MoPE融合需要使用）
        dump_attn=False,                    # 是否保存注意力图，调试用
        prompt_depth_override=None,         # 覆盖默认的Prompt深度
        prompt_embeddings=None,             # TODO 来自MoPE Router的外部Prompt向量
        prompt_length_override=None,        # 覆盖默认的Prompt长度
        vision_embedding=None,              # TODO 视觉特征，和上面的embeddings区别在哪？
        blind_prompt=False,                 # 是否屏蔽Prompt，用于消融实验
    ):
        # text_input: List[str]
        # return: logits
        # 允许在推理时动态改变 Prompt 的深度和长度
        if prompt_depth_override is not None:
            self.prompt_depth = prompt_depth_override
        if prompt_length_override is not None:
            prompt_length = prompt_length_override
        else:
            prompt_length = self.prompt_length

        tokenizer_output = self.tokenizer(                                                                  # TODO Tokenization分词，把文本转换为Tensor
            text_input, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = tokenizer_output["input_ids"].to(self.bert_encoder.device)                              # TODO 将数据搬运到GPU，什么作用？
        orig_input = self.tokenizer.decode(input_ids[0])
        token_type_ids = tokenizer_output["token_type_ids"].to(self.bert_encoder.device)                    # TODO 将数据搬运到GPU，什么作用？
        attention_mask = tokenizer_output["attention_mask"].to(self.bert_encoder.device)                    # TODO 将数据搬运到GPU，什么作用？
        # txt_attn_mask = self.get_extended_txt_attn_mask(attention_mask)
        # txt_tokens = self.bert_encoder.embeddings(input_ids, token_type_ids)
        # for bert_layer_id in range(self.bert_encoder.config.num_hidden_layers):
        #     txt_tokens = self.bert_encoder.encoder.layer[bert_layer_id](txt_tokens, attention_mask)[0]
        prompt_start_idx = 1                                                                                # Prompt插入到第0个token([CLS])之后
        # use_vision_feature = True
        # if vision_feature is not None: # assume vision feature is of same dimension as bert hidden size
        #         vision_feature = vision_feature.unsqueeze(1)
        #         prompt_start_idx  = 2
        #         use_vision_feature = True

        if not self.use_prompt:                                                                             # 关闭Prompt Tuning模式
            bert_output = self.bert_encoder(
                input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
            )
            cls_ = bert_output["pooler_output"]                                                             # 取出[CLS]位置的向量作为整句话的特征
            logits = self.heads(cls_)                                                                       # 通过分类头计算logits
        else:                                                                                               # 开启Prompt Tuning模式
            if self.prompt_depth > 0:
                attention_mask = torch.cat(                                                                 # 拼接 Mask: [CLS的Mask] + [全1的Prompt Mask] + [正文的Mask]
                    [
                        attention_mask[:, :prompt_start_idx],
                        torch.ones(attention_mask.size(0), prompt_length).to(
                            self.bert_encoder.device
                        ),
                        attention_mask[:, prompt_start_idx:],
                    ],
                    dim=1,
                )  # add one more attention mask for prompt vector
                if blind_prompt:                                                                            # 如果屏蔽Prompt，则把前面在Prompt位置上的mask修改为0
                    attention_mask[
                        prompt_start_idx + prompt_length :, :prompt_length
                    ] = 0
            txt_tokens = self.bert_encoder.embeddings(input_ids, token_type_ids)                            # 计算原始文本的embeddings（未加Prompt）
            # if use_vision_feature:
            #     #when use vision feature: input sequence is [CLS], [IMG], learnable prompt, text tokens, [SEP]
            #     txt_tokens = torch.cat([
            #         txt_tokens[:,:1,:],
            #         vision_feature,
            #         txt_tokens[:,1:,:]
            #         ],dim = 1)
            #     attention_mask = torch.cat([
            #         attention_mask[:,:1],
            #         torch.ones(attention_mask.size(0), 1).to(self.bert_encoder.device),
            #         attention_mask[:,1:]
            #         ], dim=1) # add one more attention mask for vision feature
            txt_attn_mask = self.get_extended_txt_attn_mask(attention_mask)                                 # TODO 处理Mask格式以适应Transformer计算要求，啥意思？

            for bert_layer_id in range(self.bert_encoder.config.num_hidden_layers):                         # 逐层循环Deep Prompt Tuning
                if (
                    vision_embedding is not None and prompt_embeddings is not None                          # TODO 情况A：既有视觉特征又有MoPE外部Prompt
                ):  # both are used, it means promptfuse
                    mapped_prompt = self.prompt_dropout(vision_embedding)
                    static_prompt = prompt_embeddings[bert_layer_id].squeeze(1)
                    crt_prompt_vector = torch.cat(
                        [mapped_prompt.unsqueeze(1), static_prompt], dim=1
                    )

                elif prompt_embeddings is not None:                                                         # TODO 情况B：MoPE核心，有外部MoE路由结果传入的Prompt
                    crt_prompt_vector = self.prompt_dropout(
                        prompt_embeddings[bert_layer_id]
                    )
                else:                                                                                       # TODO 情况C：默认使用内部训练的self.prompt_vector
                    crt_prompt_vector = self.prompt_dropout(
                        self.prompt_vector[bert_layer_id]
                        .unsqueeze(0)
                        .repeat(input_ids.size(0), 1, 1)
                    )
                if bert_layer_id < self.prompt_depth and self.prompt_depth > 1:                             # 如果当前层还在 Prompt 深度范围内
                    # insert prompt vector between [CLS] and the first token
                    txt_tokens = torch.cat(                                                                 # 拼接: [CLS] + [Prompt向量] + [正文]
                        [
                            txt_tokens[:, :prompt_start_idx, :],
                            crt_prompt_vector,
                            txt_tokens[:, prompt_start_idx:, :],
                        ],
                        dim=1,
                    )

                layer_output = self.bert_encoder.encoder.layer[bert_layer_id](                              # 带着Prompt通过这一层Transformer进行计算
                    txt_tokens, txt_attn_mask, output_attentions=True
                )
                if dump_attn:
                    debug_utils.dump_tensor(
                        layer_output[1][0],
                        f"bert_layer_{bert_layer_id}_attn",
                        "./debug/dump",
                    )
                txt_tokens = layer_output[0]                                                                # TODO 更新txt_token为这一层的输出，layer_output[0]是啥意思？

                # remove prompt vector
                if bert_layer_id < self.prompt_depth:                                                       # 去除当前层的prompt
                    txt_tokens = torch.cat(
                        [
                            txt_tokens[:, :prompt_start_idx, :],
                            txt_tokens[:, prompt_length + prompt_start_idx :, :],
                        ],
                        dim=1,
                    )
            cls_ = txt_tokens[:, 0, :]                                                                      # 取出第0个位置的CLS向量，充分融合了日志语义和Prompt指令
            logits = self.heads(cls_)
        if not return_features:
            return logits
        else:
            return logits, cls_



    # MoPE模块实际上走的是这个函数的“后半部分”，而前半部分主要是为了学术论文做消融实验或基准对比用的
    def forward_instruct(
        self,
        text_input,
        return_features=False,
        prompt_embeddings=None,
        prompt_length_override=None,
        vision_embedding=None,
        blind_prompt=False,
        prompt_depth_override=None,
    ):
        if (
            vision_embedding is not None
        ):  # when use vision embedding, then it means that we ablate promptfuse baseline
            self.prompt_length = 1
            B = vision_embedding.size(0)
            # init the static prompt embedding
            static_prompt_experts = [
                p[0, ...].unsqueeze(0).expand(B, -1, -1, -1)
                for p in self.prompt_vectors
            ]
            prompt_embeddings = static_prompt_experts
            prompt_length_override = static_prompt_experts[0].shape[2] + 1
            logit, cls_ = self.forward(
                text_input,
                return_features=True,
                prompt_embeddings=prompt_embeddings,
                prompt_length_override=prompt_length_override,
                vision_embedding=vision_embedding,
                blind_prompt=blind_prompt,
                prompt_depth_override=prompt_depth_override,
            )
            return logit, cls_

        return self.forward(
            text_input,
            return_features,
            prompt_embeddings=prompt_embeddings,
            prompt_length_override=prompt_length_override,
            vision_embedding=vision_embedding,
        )



    # 根据路由分数（route_score）和视觉特征（vision_input），将“静态专家”、“动态专家”和“视觉特征”混合在一起，调制出每一层 BERT 专属的 Prompt
    def forward_instruct_moe(
        self, text_input, vision_input, route_score, return_features=False
    ):
        """
        route_score: [B, n_expert] determine which expert(s) to use. if more than one, interpolate between the prompts
        [B, n_layer, n_expert] for per-layer routed moe
        """

        B = vision_input.size(0)
        if len(route_score.shape) == 2:                                                         # TODO 判断路由模式：每一层用不同权重还是共用一套权重，啥是权重？
            route_per_layer = False
        elif len(route_score.shape) == 3:
            route_per_layer = True
        y0 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_0(vision_input)))        # 把BERT分成了3个阶段，y0，y1，y2分别是针对底层、中层、高层的视觉Prompt
        y1 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_1(vision_input)))
        y2 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_2(vision_input)))
        all_prompt_experts = [                                                                  # 动态专家：prompt_vectors索引1之后的所有向量
            p[1:, ...].expand(B, -1, -1, -1) for p in self.prompt_vectors
        ]
        static_prompt_experts = [                                                               # 静态专家：prompt_vectors索引0的向量
            p[0, ...].unsqueeze(0).expand(B, -1, -1, -1) for p in self.prompt_vectors
        ]

        moe_prompt_embds = []
        for i in range(len(all_prompt_experts)):                                                # 遍历BERT的12层
            if route_per_layer:
                # b batch, k expert, l seq len, h hidden dim
                crt_prompt = torch.einsum(                                                      # TODO MoPE核心路由逻辑，没看懂
                    "bk,bklh->blh", route_score[:, i, :], all_prompt_experts[i]
                )
            else:
                crt_prompt = torch.einsum(
                    "bk,bklh->blh", route_score, all_prompt_experts[i]
                )

            # concate projected prompt
            if i < 4:
                projected_prompt = y0
            elif i < 8:
                projected_prompt = y1
            else:
                projected_prompt = y2
            projected_prompt = projected_prompt.unsqueeze(1)
            crt_prompt = torch.cat(                                                             # TODO [视觉Prompt] + [静态Prompt] + [混合后的动态Prompt]，这个视觉Prompt是论文里的Mapper吗？
                [projected_prompt, static_prompt_experts[i].squeeze(1), crt_prompt],
                dim=1,
            )  # dim 1 is l
            moe_prompt_embds.append(crt_prompt)                                                 # 将构造好的这一层的 Prompt 加入列表
        prompt_length = 2 * self.prompt_length + 1
        return self.forward_instruct(
            text_input,
            prompt_embeddings=moe_prompt_embds,
            prompt_length_override=prompt_length,
        )



    # 消融实验：将视觉特征拼接在静态prompt上，不使用动态路由机制
    def forward_seqfuse(self, text_input, vision_input, return_features=False):
        """
        route_score: [B, n_expert] determine which expert(s) to use. if more than one, interpolate between the prompts
        [B, n_layer, n_expert] for per-layer routed moe
        """

        B = vision_input.size(0)
        y0 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_0(vision_input)))
        y1 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_1(vision_input)))
        y2 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_2(vision_input)))

        static_prompt_experts = [
            p[0, ...].unsqueeze(0).expand(B, -1, -1, -1) for p in self.prompt_vectors
        ]

        moe_prompt_embds = []
        for i in range(len(static_prompt_experts)):

            if i < 4:
                projected_prompt = y0
            elif i < 8:
                projected_prompt = y1
            else:
                projected_prompt = y2
            projected_prompt = projected_prompt.unsqueeze(1)
            crt_prompt = torch.cat(
                [projected_prompt, static_prompt_experts[i].squeeze(1)],
                dim=1,
            )  # dim 1 is l
            moe_prompt_embds.append(crt_prompt)
        prompt_length = self.prompt_length + 1
        return self.forward_instruct(
            text_input,
            prompt_embeddings=moe_prompt_embds,
            prompt_length_override=prompt_length,
            return_features=True,
        )



    def freeze_backbone(self):
        """
        freeze the backbone of the model, except the prompt vector and head
        """
        for name, param in self.named_parameters():
            if "prompt" not in name and "head" not in name:
                param.requires_grad = False

        if not self.use_prompt:
            print("[Warning] Freezeing Bert backbone but without prompt vector")

    def get_extended_txt_attn_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


if __name__ == "__main__":
    text = ["Hello, my dog is cute", "Hello, my cat is cute, too"]
    model = BertClassifier(4)
    logits = model(text)
    print(logits)
    print(F.log_softmax(logits))
