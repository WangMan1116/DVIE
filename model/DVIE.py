import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DVIE(nn.Module):
    def __init__(self, config, att, init_clip_att, seenclass, unseenclass, is_bias=True, bias=1, is_conservative=True):
        super(DVIE, self).__init__()
        self.config = config
        self.dim_f_clip = config.dim_f_clip  # clip视觉特征维度768
        self.dim_v_clip = config.dim_v_clip  # clip语义分支维度
        self.nclass = config.num_class
        self.seenclass = seenclass
        self.unseenclass = unseenclass
        self.is_bias = is_bias
        self.is_conservative = is_conservative

        # 类别属性标签（专家标注） [200, 312]
        self.att = nn.Parameter(F.normalize(att), requires_grad=False)
        self.V_clip = nn.Parameter(F.normalize(init_clip_att), requires_grad=True)

        # for self-calibration
        self.bias = nn.Parameter(torch.tensor(bias), requires_grad=False)
        mask_bias = np.ones((1, self.nclass))
        mask_bias[:, self.seenclass.cpu().numpy()] *= -1
        self.mask_bias = nn.Parameter(torch.tensor(
            mask_bias, dtype=torch.float), requires_grad=False)

        # W2: 将 enhance 输出映射到语义空间
        self.W_2 = nn.Parameter(nn.init.normal_(
            torch.empty(self.dim_v_clip, config.tf_clip_common_dim)), requires_grad=True)

        # clip增强视觉特征
        self.clip_enhancer = ClipEnhanceModule(
            dim_visual=config.dim_f_clip,
            dim_text=config.dim_v_clip,
            dim_common=config.tf_clip_common_dim,
            heads=config.tf_heads
        )

        self.log_softmax_func = nn.LogSoftmax(dim=1)
        self.weight_ce = nn.Parameter(torch.eye(self.nclass), requires_grad=False)

        # self.visual_to_att = nn.Linear(config.dim_f_clip, config.num_attribute)

    def forward(self, clip_input):
        # === 修改说明 ===
        # clip_input 现在的形状是 [B, N_patches, 768] (例如 [50, 577, 768])
        # 而不是之前的 [B, 768]
        clip_feat = clip_input
        V_n_clip = F.normalize(self.V_clip) if self.config.normalize_V else self.V_clip  # [312，768]
        # 1.空间注意力增强
        clip_out,attn_weights  = self.clip_enhancer(clip_feat, V_n_clip)  # [50, 300]
        # 保存或缓存到 self，便于后续可视化使用
        self.last_attn_weights = attn_weights  # [B, 312]
        # 2. 语义映射
        V_proj = torch.matmul(V_n_clip, self.W_2)  # [312, 300]
        # 3. 生成视觉嵌入 [B, 312]
        clip_embed = torch.matmul(clip_out, V_proj.T)
        # 4. 分类 Logits
        clip_logits = torch.einsum('ki,bi->bk', self.att, clip_embed)
        self.vec_bias = self.mask_bias * self.bias
        clip_logits = clip_logits + self.vec_bias
        package = {
            'clip_pred': clip_logits,
            'clip_embed': clip_embed,
            'clip_S_pp': clip_logits
        }

        # 基线模型
        # pred_att = self.visual_to_att(clip_input)
        # pred_att_norm = F.normalize(pred_att, dim=1)  # [B, 312]
        # att_norm = F.normalize(self.att, dim=1)
        # clip_logits = torch.matmul(pred_att_norm, att_norm.t())
        # self.vec_bias = self.mask_bias * self.bias
        # package = {
        #     'clip_pred': clip_logits,
        #     'clip_embed': clip_feat,
        # }
        # package['clip_S_pp'] = package['clip_pred']
        # return package

    # 分类交叉熵
    def compute_aug_cross_entropy(self, in_package):
        Labels = in_package['batch_label']  # [50]
        clip_S_pp = in_package['clip_pred']  # (50,200)
        if self.is_bias:
            clip_S_pp = clip_S_pp - self.vec_bias  # 处理偏置
        if not self.is_conservative:  # 如果 is_conservative=False，只考虑已见类别 seenclass
            clip_S_pp = clip_S_pp[:, self.seenclass]
            Labels = Labels[:, self.seenclass]
            assert clip_S_pp.size(1) == len(self.seenclass)

        clip_Prob = self.log_softmax_func(clip_S_pp)  # 计算 log_softmax 概率

        # 计算增强的交叉熵损失
        loss = -torch.einsum('bk,bk->b', clip_Prob, Labels)
        loss = torch.mean(loss)
        return loss

    # 属性回归
    def compute_reg_loss(self, in_package):
        tgt = torch.matmul(in_package['batch_label'], self.att)
        clip_embed = in_package['clip_embed']
        loss_reg = F.mse_loss(clip_embed, tgt, reduction='mean')
        return loss_reg

    # 自校准
    def compute_loss_Self_Calibrate(self, in_package):
        S_pp = in_package['clip_pred']
        Prob_all = F.softmax(S_pp, dim=-1)
        self.unseenclass = self.unseenclass.long()
        Prob_unseen = Prob_all[:, self.unseenclass]
        assert Prob_unseen.size(1) == len(self.unseenclass)
        mass_unseen = torch.sum(Prob_unseen, dim=1)
        loss_pmp = -torch.log(torch.mean(mass_unseen))
        return loss_pmp

    # 双向监督对比学习
    '''
        通过 视觉 → 语义 和 语义 → 视觉 的双向映射来拉近图像和其对应属性之间的距离，同时远离其他类别。
        输出：
            clip_embed：[B, 312]：视觉特征经过语义对齐后的嵌入
            self.att: [200, 312]：归一化后的专家属性表示（每类一个）。
            labels: [B]：图像对应的类别索引，Long 类型。
        视觉到属性：  
    '''
    def compute_contrastive_loss(self, in_package, temperature=0.05):
        # clip_embed: [B, 312]
        clip_embed = F.normalize(in_package['clip_embed'], dim=1)  # [B, 312]
        att = F.normalize(self.att, dim=1)  # [200, 312]

        # 真实标签: shape [B]，必须是 LongTensor，不是 one-hot！
        labels = in_package['batch_label']  # [B]
        if labels.ndim > 1:
            labels = torch.argmax(labels, dim=1)  # 从 one-hot 转换为索引

        # Forward: clip_embed → att     每个视觉嵌入与200个属性向量计算余弦相似度（归一化后点积）
        # 用 GT label 去做分类，即：图像的视觉嵌入 clip_embed 应该匹配上它的 att[class_id]。
        logits_vs = torch.matmul(clip_embed, att.T) / temperature  # [B, 200]
        loss_vs = F.cross_entropy(logits_vs, labels)

        # Backward: att → clip_embed    取每个图像的 Ground Truth 属性向量 gt_att，与所有图像的视觉嵌入计算相似度。
        # 监督目标是该属性向量应该最相似它所属的那个图像。
        gt_att = att[labels]  # [B, 312]
        logits_sv = torch.matmul(gt_att, clip_embed.T) / temperature  # [B, B]
        targets = torch.arange(clip_embed.size(0)).to(clip_embed.device)
        loss_sv = F.cross_entropy(logits_sv, targets)

        # 平均视觉 → 属性 和 属性 → 视觉两个方向的监督损失。
        # return loss_vs
        # return loss_sv
        return (loss_vs + loss_sv) / 2

    def compute_loss(self, in_package):
        if len(in_package['batch_label'].size()) == 1:
            in_package['batch_label'] = self.weight_ce[in_package['batch_label']]  # 如果 batch_label 是索引（(batch,)），转换为 one-hot
        loss_CE = self.compute_aug_cross_entropy(in_package)        # 交叉熵损失
        loss_reg = self.compute_reg_loss(in_package)                # 属性预测
        loss_cal = self.compute_loss_Self_Calibrate(in_package)     # 自校准
        loss_con = self.compute_contrastive_loss(in_package)        # 对比学习
        #
        loss = (
            loss_CE
            +self.config.lambda_reg * loss_reg
            +self.config.lambda_cal * loss_cal
            +self.config.lambda_con * loss_con
        )


        out_package = {
            'loss': loss,
            'loss_CE': loss_CE,
            'loss_reg':loss_reg,
            'loss_cal':loss_cal,
            'loss_con':loss_con
        }
        return out_package


'''
    这个模块让每张图像“主动学习应该关注哪些语义属性”，输出的是一种“语义感知后的视觉表示”
'''

# 先降维在注意力
class ClipEnhanceModule(nn.Module):
    def __init__(self, dim_visual=768, dim_text=768, dim_common=300, heads=4):  #单头注意力
        super(ClipEnhanceModule,self).__init__()
        self.query_proj = nn.Linear(dim_visual, dim_common)
        self.key_proj = nn.Linear(dim_text, dim_common)
        self.value_proj = nn.Linear(dim_text, dim_common)

        self.attn = nn.MultiheadAttention(dim_common, num_heads=heads, batch_first=True)    # 视觉特征与语义词向量交互，捕获属性线索
        self.norm = nn.LayerNorm(dim_common)
        self.fc_out = nn.Linear(dim_common, dim_common)

    def forward(self, clip_feat, init_clip_att):
        """
        clip_feat: [B, 768]
        clip_wordvec: [312, 768]
        """
        B = clip_feat.size(0)

        # 1. 映射 Query (Visual)
        # [B, N, 768] -> [B, N, dim_common]
        query = self.query_proj(clip_feat)
        # 2. 映射 Key/Value (Semantic Attributes)
        # [312, 768] -> [B, 312, dim_common]
        key = self.key_proj(init_clip_att).unsqueeze(0).expand(B, -1, -1)
        value = self.value_proj(init_clip_att).unsqueeze(0).expand(B, -1, -1)
        # 3. 空间注意力交互
        # 输出: attn_out [B, N, dim_common]
        # 这里的意义是：对于图像的每一个 Patch，我们都根据属性进行增强
        attn_out, attn_weights = self.attn(query, key, value, need_weights=True)
        # 4. 残差连接 + Norm
        out = self.norm(attn_out + query)
        # 5. 特征变换
        enhance_map = self.fc_out(out)  # [B, N, dim_common]
        # 6. === 关键改动：空间特征聚合 ===
        # 我们有 577 个特征向量，需要融合成 1 个代表整张图的向量。
        # 方法 A: 取第一个 (Class Token) -> enhance_map[:, 0, :]
        # 方法 B: 全局平均池化 (GAP) -> enhance_map.mean(dim=1)
        # 这里推荐使用 GAP，因为它利用了所有空间位置的信息。
        enhance_out = enhance_map.mean(dim=1)  # [B, dim_common]
        return enhance_out,attn_weights.squeeze(1)

# 先注意力再降维
# class ClipEnhanceModule(nn.Module):
#     def __init__(self, dim_visual=768, dim_text=768, dim_common=300, heads=4):
#         super(ClipEnhanceModule, self).__init__()
#         # 注意力仍在 768 维空间里执行
#         self.query_proj = nn.Linear(dim_visual, dim_visual)
#         self.key_proj = nn.Linear(dim_text, dim_text)
#         self.value_proj = nn.Linear(dim_text, dim_text)
#
#         self.attn = nn.MultiheadAttention(dim_visual, num_heads=heads, batch_first=True)
#
#         self.norm = nn.LayerNorm(dim_visual)
#         # 最后统一映射到公共语义空间 300 维
#         self.fc_out = nn.Linear(dim_visual, dim_common)
#
#     def forward(self, clip_feat, init_clip_att):
#         """
#         clip_feat: [B, 768]
#         init_clip_att: [312, 768]
#         """
#         B = clip_feat.size(0)
#         clip_feat = clip_feat.unsqueeze(1)  # [B, 1, 768]
#
#         # 先在 768 维做 query/key/value
#         query = self.query_proj(clip_feat)  # [B, 1, 768]
#         key = self.key_proj(init_clip_att).unsqueeze(0).expand(B, -1, -1)  # [B, 312, 768]
#         value = self.value_proj(init_clip_att).unsqueeze(0).expand(B, -1, -1)  # [B, 312, 768]
#
#         # 注意力交互（发生在 768 维）
#         attn_out, attn_weights = self.attn(query, key, value, need_weights=True)  # [B, 1, 768]
#
#         # 残差 + LN
#         out = self.norm(attn_out + query)
#
#         # 最后映射到公共空间 (300维)
#         enhance_out = self.fc_out(out).squeeze(1)  # [B, 300]
#
#         return enhance_out, attn_weights.squeeze(1)
