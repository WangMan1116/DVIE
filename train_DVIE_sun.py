import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import yaml
import clip
from types import SimpleNamespace
from model.DVIE import DVIE
# [修改点 1] 引入 SUN 的 DataLoader
from tools.dataset import SUNDataLoader
from tools.helper_func import eval_zs_gzsl

# 加载配置项
# [修改点 2] 加载 SUN 的配置文件 (请确认文件名是否正确)
config_path = 'config/sun_gzsl.yaml'
try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"Error: 找不到配置文件 {config_path}，请检查文件名。")
    exit()

config = {
    k: v['value'] if isinstance(v, dict) and 'value' in v else v
    for k, v in config.items()
}
config = SimpleNamespace(**config)





print("Loaded Configuration for SUN:")
print(config)

# 确保配置里有 device
if not hasattr(config, 'device'):
    config.device = 'cuda:0'

print("Initializing Training...")

# 1. 加载 CLIP Backbone (ViT-L/14@336px)
print(f"Loading CLIP model: ViT-L/14@336px ...")
clip_model, _ = clip.load("ViT-L/14@336px", device=config.device)
clip_model.eval()  # 冻结参数，只做特征提取
for param in clip_model.parameters():
    param.requires_grad = False

# 加载数据集
# [修改点 3] 实例化 SUNDataLoader
print("Loading SUN Dataset...")
dataloader = SUNDataLoader('.', config.device, is_balance=False)

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

    # 在线提取 CLIP 特征
    with torch.no_grad():
        clip_features = clip_model.encode_image(batch_images).float().to(config.device)

    # 前向传播
    out_package = model(clip_features)

    # 保存 label 以用于计算损失
    in_package = out_package
    in_package['batch_label'] = batch_label

    # 计算损失
    out_package = model.compute_loss(in_package)

    # 提取各项 Loss
    loss, loss_CE, loss_cal, loss_reg, loss_con = out_package['loss'], out_package['loss_CE'], out_package['loss_cal'], out_package['loss_reg'], \
    out_package['loss_con']

    loss.backward()
    optimizer.step()

    # 打印评估结果
    if i % report_interval == 0:
        print('-' * 30)
        # 将 clip_model 传给 eval 函数，用于测试集的实时特征提取
        acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(dataloader, clip_model, model, config.device)

        if H > best_performance[2]:
            best_performance = [acc_novel, acc_seen, H, acc_zs]
        if acc_zs > best_performance_zsl:
            best_performance_zsl = acc_zs
            # [可选] 如果你想保存模型，建议把文件名里的 CUB 改成 SUN
            # save_path = '../Ablation_Study/save_model/SUN_best_model_zsl.pth'
            # torch.save(model.state_dict(), save_path)

        print('epoch=%d | '
              'loss=%.3f, loss_CE=%.3f, loss_cal=%.3f, loss_reg=%.3f, loss_con=%.3f| '
              'acc_unseen=%.3f, acc_seen=%.3f, H=%.3f | acc_zs=%.3f | ' % (
                  int(i // report_interval),
                  loss.item(), loss_CE.item(), loss_cal.item(), loss_reg.item(), loss_con.item(),
                  best_performance[0], best_performance[1], best_performance[2], best_performance_zsl
              ))