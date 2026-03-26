import torch
import torch.optim as optim
import numpy as np
import yaml
import clip
from types import SimpleNamespace
from model.DVIE import DVIE

# [修改点 1] 引入 AWA2 的 DataLoader
# 请确保 tools/dataset.py 中已经添加了 AWA2DataLoader 类
from tools.dataset import AWA2DataLoader
from tools.helper_func import eval_zs_gzsl

# [修改点 2] 加载 AWA2 的配置文件
config_path = './config/clip_branch_awa2_gzsl.yaml'

try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"Error: 找不到配置文件 {config_path}。")
    print("请先创建 config/clip_banch_awa2_gzsl.yaml 文件。")
    exit()

# 将 yaml 字典转换为 SimpleNamespace 对象
config = {
    k: v['value'] if isinstance(v, dict) and 'value' in v else v
    for k, v in config.items()
}
config = SimpleNamespace(**config)

print("Loaded Configuration for AWA2:")
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

# [修改点 3] 实例化 AWA2DataLoader
print("Loading AWA2 Dataset...")
# 注意：AWA2DataLoader 内部会自动处理 JPEGImages 路径
dataloader = AWA2DataLoader('.', config.device, is_balance=False)

# 固定随机种子
seed = config.random_seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# 初始化 DVIE 模型
# AWA2 的 seenclasses 和 unseenclasses 数量与 SUN 不同 (40/10 vs 645/72)
# AWA2 的属性维度是 85 (SUN 是 102)
# 这些都在 dataloader 中自动处理了，传给模型即可
model = DVIE(config, dataloader.att, dataloader.clip_att, dataloader.seenclasses, dataloader.unseenclasses).to(config.device)

# 设置优化器
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

# 训练参数
niters = dataloader.ntrain_clip * config.epochs // config.batch_size
report_interval = niters // config.epochs
best_performance = [0, 0, 0, 0] # acc_novel, acc_seen, H, acc_zs
best_performance_zsl = 0

print(f"Start training loop for {niters} iterations...")

# 主训练循环
for i in range(0, niters):
    model.train()
    optimizer.zero_grad()

    try:
        # 获取图片 Batch (Label, Image, Att)
        batch_label, batch_images, batch_att = dataloader.next_batch(config.batch_size)
    except Exception as e:
        print(f"Skipping batch {i} due to dataloader error: {e}")
        continue

    # 在线提取 CLIP 特征
    with torch.no_grad():
        clip_features = clip_model.encode_image(batch_images).float().to(config.device)

    # 前向传播
    out_package = model(clip_features)

    # 保存 label 以用于计算损失
    # 建议使用 .copy() 以避免潜在的引用问题，虽然直接赋值在简单循环中通常也没问题
    in_package = out_package.copy()
    in_package['batch_label'] = batch_label

    # 计算损失
    out_package = model.compute_loss(in_package)

    # 提取各项 Loss
    loss = out_package['loss']
    loss_CE = out_package['loss_CE']
    loss_cal = out_package['loss_cal']
    loss_reg = out_package['loss_reg']
    loss_con = out_package['loss_con']

    loss.backward()
    optimizer.step()

    # 打印评估结果
    if i % report_interval == 0:
        print('-' * 30)
        # 评估
        # 提示：AWA2 有时需要调整 bias 才能获得最佳的 H 值
        # 这里默认 bias=0，如果 seen 很高但 novel 很低，可以尝试传入 bias_seen=-0.2 等参数
        acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(dataloader, clip_model, model, config.device)

        if H > best_performance[2]:
            best_performance = [acc_novel, acc_seen, H, acc_zs]
            # 保存最佳模型
            # save_path = f'./save_model/AWA2_best_H_{H:.3f}.pth'
            # torch.save(model.state_dict(), save_path)

        if acc_zs > best_performance_zsl:
            best_performance_zsl = acc_zs

        print('epoch=%d | '
              'loss=%.3f, CE=%.3f, cal=%.3f, reg=%.3f, con=%.3f| '
              'U=%.3f, S=%.3f, H=%.3f | ZS=%.3f' % (
                  int(i // report_interval),
                  loss.item(), loss_CE.item(), loss_cal.item(), loss_reg.item(), loss_con.item(),
                  best_performance[0], best_performance[1], best_performance[2], best_performance_zsl
              ))