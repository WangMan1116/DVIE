import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import yaml
import clip
from types import SimpleNamespace
from model.DVIE import DVIE
from tools.dataset import CUBDataLoader
from tools.helper_func import eval_zs_gzsl

# 测试工作路径
# import os
# current_directory = os.getcwd()
# print(current_directory)      #/data/wm/DVIE_modify

# === 新增函数：提取 CLIP 空间特征 ===
def get_clip_spatial_features(clip_model,images):
    """
        手动执行 CLIP Visual Encoder 的前向传播，
        返回形状为 [B, N_tokens, Dim] 的特征，而不是 [B, Dim]。
        N_tokens = (H/14 * W/14) + 1.
        对于 3316px 输入，N = 24*24+1 = 577.
    """
    with torch.no_grad():
        x = clip_model.visual.conv1(images) # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # 添加 Class Token
        class_embedding = clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                                      dtype=x.dtype, device=x.device)
        x = torch.cat([class_embedding, x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # 添加位置编码
        x = x + clip_model.visual.positional_embedding.to(x.dtype)
        x = clip_model.visual.ln_pre(x)
        # Transformer 层
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = clip_model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        # Post Norm
        x = clip_model.visual.ln_post(x)
        # 投影层 (ViT-L/14 有 proj: 1024 -> 768)
        if clip_model.visual.proj is not None:
            x = x @ clip_model.visual.proj

    return x # 返回 [B, 577, 768]

# 加载配置项
with open('./config/cub_gzsl.yaml', 'r') as f:
    config = yaml.safe_load(f)
config = {
    k: v['value'] if isinstance(v, dict) and 'value' in v else v
    for k, v in config.items()
}
config = SimpleNamespace(**config)
print(config)
# 确保配置里有 device
if not hasattr(config, 'device'):
    config.device = 'cuda:0'

print("Initializing Training...")

# 1. 加载 CLIP Backbone (ViT-L/14@336px)
# 注意：ViT-L/14@336px 在 clip 库中的名字通常是 "ViT-L/14@336px"
print(f"Loading CLIP model: ViT-L/14@336px ...")
clip_model, _ = clip.load("ViT-L/14@336px", device=config.device)
clip_model = clip_model.float()
clip_model.eval()  # 冻结参数，只做特征提取
for param in clip_model.parameters():
    param.requires_grad = False

# 加载数据集
print("Loading CUB Dataset...")
dataloader = CUBDataLoader('.', config.device, is_balance=False)

# 固定随机种子
seed = config.random_seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# 初始化DVIE模型
model = DVIE(config, dataloader.att, dataloader.clip_att, dataloader.seenclasses, dataloader.unseenclasses).to(config.device)

# 设置优化器
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

# 训练参数
niters = dataloader.ntrain_clip * config.epochs // config.batch_size
report_interval = niters // config.epochs
best_performance = [0, 0, 0, 0]
best_performance_zsl = 0
best_performance_H = 0

print(f"Start training loop for {niters} iterations...")
# 主训练循环
for i in range(0, niters):
    model.train()
    optimizer.zero_grad()

    # 获取图片 Batch (Label, Image, Att)
    batch_label, batch_images, batch_att = dataloader.next_batch(config.batch_size)

    # === 修改点：使用自定义的空间特征提取函数 ===
    clip_features = get_clip_spatial_features(clip_model, batch_images).float().to(config.device)

    # # 5. 在线提取 CLIP 特征 之前的DVIE使用全局特征
    # with torch.no_grad():
    #     clip_features = clip_model.encode_image(batch_images).float().to(config.device)

    # 前向传播
    out_package = model(clip_features)

    # 保存 label 以用于计算损失
    in_package = out_package
    in_package['batch_label'] = batch_label # [50]
    # 计算损失
    out_package = model.compute_loss(in_package)
    # loss, loss_CE = out_package['loss'], out_package['loss_CE']
    # loss, loss_CE, loss_con= out_package['loss'], out_package['loss_CE'], out_package['loss_con']
    # loss, loss_CE, loss_con, loss_reg= out_package['loss'], out_package['loss_CE'], out_package['loss_con'], out_package['loss_reg']
    loss, loss_CE, loss_cal, loss_reg, loss_con = out_package['loss'], out_package['loss_CE'], out_package['loss_cal'], out_package['loss_reg'], out_package['loss_con']
    loss.backward()
    optimizer.step()

    # 打印评估结果
    if i % report_interval == 0:
        print('-'*30)
        # 将 clip_model 传给 eval 函数，用于测试集的实时特征提取
        acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(dataloader, clip_model, model, config.device)

        if H > best_performance[2]:
            best_performance = [acc_novel, acc_seen, H, acc_zs]
        if acc_zs > best_performance_zsl:
            best_performance_zsl = acc_zs
#             best_performance_H = H
#             save_path = '../Ablation_Study/save_model/CUB_best_model_zsl.pth'  # 修改为你想要保存的路径
#             torch.save(model.state_dict(), save_path)
#             print(f"=> Best ZSL model saved to {save_path}")
#             # print('epoch=%d | '
#             #       'loss=%.3f, loss_CE=%.3f| '
#             #       'acc_unseen=%.3f, acc_seen=%.3f, H=%.3f | acc_zs=%.3f | ' % (
#             #     int(i // report_interval),
#             #     loss.item(), loss_CE.item(),
#             #     best_performance[0], best_performance[1],best_performance[2], best_performance_zsl
#             # ))
#
#             print('epoch=%d | '
#                   'loss=%.3f, loss_CE=%.3f, loss_con=%.3f| '
#                   'acc_unseen=%.3f, acc_seen=%.3f, H=%.3f | acc_zs=%.3f | ' % (
#                       int(i // report_interval),
#                       loss.item(), loss_CE.item(), loss_con.item(),
#                       best_performance[0], best_performance[1], best_performance[2], best_performance_zsl
#                   ))
#
#             # print('epoch=%d | '
#             #       'loss=%.3f, loss_CE=%.3f, loss_con=%.3f, loss_reg=%.3f| '
#             #       'acc_unseen=%.3f, acc_seen=%.3f, H=%.3f | acc_zs=%.3f | ' % (
#             #           int(i // report_interval),
#             #           loss.item(), loss_CE.item(), loss_con.item(), loss_reg.item(),
#             #           best_performance[0], best_performance[1], best_performance[2], best_performance_zsl
#             #       ))
#
        print('epoch=%d | '
              'loss=%.3f, loss_CE=%.3f, loss_cal=%.3f, loss_reg=%.3f, loss_con=%.3f| '
              'U=%.3f, S=%.3f, H=%.3f | ZS=%.3f | ' % (
            int(i // report_interval),
            loss.item(), loss_CE.item(), loss_cal.item(), loss_reg.item(), loss_con.item(),
            best_performance[0], best_performance[1],best_performance[2], best_performance_zsl
        ))
#
#         # print('iter/epoch=%d/%d | '
#         #       'loss=%.3f, loss_CE=%.3f| '
#         #       'acc_unseen=%.3f, acc_seen=%.3f, H=%.3f | acc_zs=%.3f' % (
#         #     i, int(i // report_interval),
#         #     loss.item(),loss_CE.item(),
#         #     best_performance[0], best_performance[1],best_performance[2], best_performance_zsl))
#
#         print('epoch=%d | '
#               'loss=%.3f, loss_CE=%.3f, loss_con=%.3f| '
#               'acc_unseen=%.3f, acc_seen=%.3f, H=%.3f | acc_zs=%.3f | ' % (
#                   int(i // report_interval),
#                   loss.item(), loss_CE.item(), loss_con.item(),
#                   best_performance[0], best_performance[1], best_performance[2], best_performance_zsl
#               ))
#
#         # print('epoch=%d | '
#         #       'loss=%.3f, loss_CE=%.3f, loss_con=%.3f, loss_reg=%.3f| '
#         #       'acc_unseen=%.3f, acc_seen=%.3f, H=%.3f | acc_zs=%.3f | ' % (
#         #           int(i // report_interval),
#         #           loss.item(), loss_CE.item(), loss_con.item(), loss_reg.item(),
#         #           best_performance[0], best_performance[1], best_performance[2], best_performance_zsl
#         #       ))
#
#         # print('iter/epoch=%d/%d | '
#         #       'loss=%.3f, loss_CE=%.3f, loss_cal=%.3f, loss_reg=%.3f, loss_con=%.3f| '
#         #       'acc_unseen=%.3f, acc_seen=%.3f, H=%.3f | acc_zs=%.3f' % (
#         #           i, int(i // report_interval),
#         #           loss.item(), loss_CE.item(), loss_cal.item(), loss_reg.item(), loss_con.item(),
#         #           best_performance[0], best_performance[1], best_performance[2], best_performance_zsl))
