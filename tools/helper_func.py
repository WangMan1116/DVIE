import torch
import torch.nn.functional as F

def get_clip_spatial_features(clip_model, images):
    with torch.no_grad():
        # 卷积层切块 卷积核大小14x14 步长14 在336x336的图上滑动，切出576块
        x = clip_model.visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        # 构造CLS Token，图像块特征拼接
        class_embedding = clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                                      dtype=x.dtype, device=x.device)
        # 将CLS Token拼接到图像块的前面 “火车头”
        x = torch.cat([class_embedding, x], dim=1)
        # 位置编码，定位每块在哪儿
        x = x + clip_model.visual.positional_embedding.to(x.dtype)
        x = clip_model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        # 经过Transformer 全局交互，上下文感知
        x = clip_model.visual.transformer(x)
        x = x.permute(1, 0, 2)
        x = clip_model.visual.ln_post(x)
        # 投影到768维空间，可以直接跟我的语义去交互
        if clip_model.visual.proj is not None:
            x = x @ clip_model.visual.proj
    # 返回的x包括全局特征和局部特征   全局：x[:, 0, :]   局部：x[:, 1:, :]
    # global_feat = x[:, 0, :] 形状[64, 768]  local_feat = x[:, 1:, :] 形状[64, 576, 768]
    return x

# acc_seen = val_gzsl(test_seen_feature, test_seen_label, seenclasses, in_package,bias=bias_seen)
# 核心：在线特征提取与预测
def extract_and_predict(loader, clip_model, model, device, target_classes=None, bias=0, is_zsl=False):
    model.eval()
    clip_model.eval()  # 确保 CLIP 也在 eval 模式
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            # 1. 实时通过 CLIP 提取特征 全局特征
            # encode_image 输出 [B, 768] (如果是 ViT-L/14)
            # features = clip_model.encode_image(batch_images).float()

            # === 修改点 ===
            features = get_clip_spatial_features(clip_model, batch_images).float()

            # 2. 将特征输入你的 DVIE 模型
            out_package = model(features)
            output = out_package['clip_S_pp']
            # 3. 处理预测 (ZSL vs GZSL)
            if is_zsl:
                # ZSL: 仅在 Unseen 类中比较
                output_t = output.clone()
                # 这里的逻辑是: 把非 Unseen 的类分数设为极小，或者只取 Unseen 的列
                # 原代码通常是给 Target Classes 加分，或者Mask掉其他类
                # 这里采用 Mask 逻辑：只看 Unseen 列
                # 注意：target_classes 必须是 Unseen Classes 的索引
                pred = torch.argmax(output_t.data[:, target_classes], 1)
                # 注意：这里 pred 返回的是 0~49 的相对索引，需要映射回全局索引用于对比？
                # 通常 acc_zs 是在 50 类内部算的，所以 label 也需要映射
            else:
                # GZSL: Seen + Unseen
                if target_classes is not None:
                    output[:, target_classes] = output[:, target_classes] + bias
                pred = torch.argmax(output.data, 1)

            predicted_labels.append(pred.cpu())
            true_labels.append(batch_labels.cpu())
    return torch.cat(true_labels), torch.cat(predicted_labels)


def eval_zs_gzsl(dataloader, clip_model, model, device, bias_seen=0, bias_unseen=0):
    model.eval()  # 评估模式

    # 所有已见/未见类别索引
    seenclasses = dataloader.seenclasses
    unseenclasses = dataloader.unseenclasses.long()

    in_package = {'model': model, 'device': device}
    '''
            见类的准确率 acc_seen      衡量模型对已见类别的分类准确率
            未见类的准确率 acc_novel   衡量模型在广义零样本学习（GZSL）下未见类别的分类表现
            零样本准确率 acc_zs        仅针对零样本学习（ZSL）
    '''
    with torch.no_grad():
        # 计算 Seen 类的准确率 (GZSL)
        # 我们需要重写 val_gzsl 来支持 loader + clip_model
        acc_seen = val_gzsl_online(dataloader.test_seen_loader, clip_model, model, seenclasses, in_package, bias=bias_seen)

        # 计算 Unseen 类的准确率 (GZSL 和 ZSL)
        acc_novel, acc_zs = val_zs_gzsl_online(dataloader.test_unseen_loader, clip_model, model, unseenclasses, in_package, bias=bias_unseen)

    if (acc_seen + acc_novel) > 0:
        H = (2 * acc_seen * acc_novel) / (acc_seen + acc_novel)
    else:
        H = 0
    return acc_seen, acc_novel, H, acc_zs


def map_label(label, classes):
    # 创建一个与 label 形状相同的张量 mapped_label，初始值全部设为 -1
    mapped_label = torch.LongTensor(label.size()).fill_(-1)
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    # 将 label 对应到 unseen_classes 的新索引
    return mapped_label


def val_gzsl_online(loader, clip_model, model, target_classes, in_package, bias=0):
    device = in_package['device']
    true_labels, predicted_labels = extract_and_predict(loader, clip_model, model, device, target_classes, bias, is_zsl=False)

    # 将 Tensor 转到 device 上计算准确率
    true_labels = true_labels.to(device)
    predicted_labels = predicted_labels.to(device)
    return compute_per_class_acc_gzsl(true_labels, predicted_labels, target_classes, in_package)


def val_zs_gzsl_online(loader, clip_model, model, unseen_classes, in_package, bias=0):
    device = in_package['device']
    clip_model.eval()
    model.eval()

    pred_gzsl_list = []
    pred_zs_t_list = []
    true_labels_list = []

    with torch.no_grad():
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            # 提取特征 全局特征
            # features = clip_model.encode_image(batch_images).float()
            # 此处修改，提取空间特征图
            features = get_clip_spatial_features(clip_model, batch_images).float()
            # 模型推理
            out_package = model(features)
            output = out_package['clip_S_pp']

            # --- ZSL (T) ---
            # 只在 Unseen 类别中找最大值
            pred_zs_t = torch.argmax(output.data[:, unseen_classes], 1)

            # --- GZSL ---
            output[:, unseen_classes] = output[:, unseen_classes] + bias
            pred_gzsl = torch.argmax(output.data, 1)

            pred_gzsl_list.append(pred_gzsl.cpu())
            pred_zs_t_list.append(pred_zs_t.cpu())
            true_labels_list.append(batch_labels.cpu())

    true_labels = torch.cat(true_labels_list).to(device)
    predicted_label_gzsl = torch.cat(pred_gzsl_list).to(device)
    predicted_label_zs_t = torch.cat(pred_zs_t_list).to(device)

    # GZSL Unseen Accuracy
    acc_gzsl = compute_per_class_acc_gzsl(true_labels, predicted_label_gzsl, unseen_classes, in_package)

    # ZSL Accuracy (需要映射 Label)
    mapped_true_labels = map_label(true_labels, unseen_classes)
    acc_zs_t = compute_per_class_acc(mapped_true_labels, predicted_label_zs_t, unseen_classes.size(0))

    return acc_gzsl, acc_zs_t


def compute_per_class_acc(test_label, predicted_label, nclass):
    test_label = test_label.to(predicted_label.device)
    acc_per_class = torch.FloatTensor(nclass).fill_(0)
    for i in range(nclass):
        idx = (test_label == i)  # 选择当前类别 i 的样本
        if idx.sum() > 0:
            acc_per_class[i] = torch.sum(test_label[idx] == predicted_label[idx]).float() / torch.sum(idx).float()
    return acc_per_class.mean().item()


def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes, in_package):
    device = in_package['device']
    per_class_accuracies = torch.zeros(target_classes.size()[0]).float().to(device).detach()  # 用于存储每个类别的准确率
    # 确保预测类别在正确的计算设备上
    predicted_label = predicted_label.to(device)

    for i in range(target_classes.size()[0]):  # 遍历所有目标类别
        is_class = test_label == target_classes[i]  # 选出该类别的样本索引
        if is_class.sum() > 0:
            per_class_accuracies[i] = torch.div((predicted_label[is_class] == test_label[is_class]).sum().float(), is_class.sum().float())
    return per_class_accuracies.mean().item()
