import os, sys, torch, pickle
import numpy as np
import scipy.io as sio
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


# 按需读取图片
class ImgDataset(Dataset):
    def __init__(self, image_files, labels, dataset_name, transform=None, root_dir=None):
        """
            Args:
                image_files: 图片路径列表 (来自 .mat 文件)
                labels: 图片对应的标签
                dataset_name: 数据集名称 ('CUB', 'SUN', 'AWA2') -> 决定路径处理逻辑
                transform: 图片预处理
                root_dir: 数据集根目录
        """
        self.image_files = image_files
        self.labels = labels
        self.dataset_name = dataset_name  # 接收传入的数据集名称
        self.transform = transform
        self.root_dir = root_dir

    # 数据集总长度，多少张图片
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1. 获取原始路径字符串
        img_path_raw = self.image_files[idx]
        if isinstance(img_path_raw, np.ndarray):
            img_path_raw = img_path_raw[0]

        # 2. 根据数据集名称，清洗路径 (模拟 preprocessing.py 的 split 逻辑，但更鲁棒)
        parts = img_path_raw.split('/')
        rel_path = img_path_raw

        if self.dataset_name == 'CUB':
            # CUB 结构通常是: root/images/001.Black_footed_Albatross/...
            if 'images' in parts:
                idx_start = parts.index('images')
                rel_path = '/'.join(parts[idx_start:])
            else:
                # 兼容旧逻辑 split_idx=6
                # 假设 raw path 是 absolute path，尝试直接拼接最后几段
                # 保底逻辑：参考 preprocessing.py 中的 CUB split_idx = 6
                # 防止路径中没有 'images' 关键字
                if len(parts) > 6:
                    rel_path = '/'.join(parts[6:])

        elif self.dataset_name == 'AWA2':
            # AWA2 结构通常是: root/JPEGImages/antelope/...
            if 'JPEGImages' in parts:
                idx_start = parts.index('JPEGImages')
                rel_path = '/'.join(parts[idx_start:])
            else:
                # 保底逻辑：参考 preprocessing.py 中的 AWA2 split_idx = 5
                if len(parts) > 5:
                    rel_path = '/'.join(parts[5:])

        elif self.dataset_name == 'SUN':
            # SUN 结构通常是: root/images/a/abbey/...
            if 'images' in parts:
                idx_start = parts.index('images')
                rel_path = '/'.join(parts[idx_start:])
            else:
                # 保底逻辑：参考 preprocessing.py 中的 SUN split_idx = 7
                if len(parts) > 7:
                    rel_path = '/'.join(parts[7:])

        # 3. 拼接完整路径
        # 这里的 root_dir 应该是 'data/CUB' 或 'data/AWA2'
        full_path = os.path.join(self.root_dir, rel_path)

        # 4. 容错加载
        try:
            image = Image.open(full_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load {full_path}, using black image.")
            image = Image.new('RGB', (336, 336))

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

class CUBDataLoader():
    def __init__(self, data_path, device, is_scale=False, is_unsupervised_attr=False, is_balance=True):
        self.data_path = data_path
        self.device = device
        self.dataset = 'CUB'
        self.root_dir = os.path.join(self.data_path, 'data', self.dataset)
        self.is_balance = is_balance
        self.is_unsupervised_attr = is_unsupervised_attr

        # 打印 数据集的根目录路径
        print(f"Loading metadata from {self.root_dir}...")

        # 1. 定义训练集增强 (Data Augmentation)
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(336, scale=(0.6, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        # 2. 定义测试集预处理
        self.test_transform = transforms.Compose([
            transforms.Resize((336, 336), interpolation=Image.BICUBIC),
            transforms.CenterCrop(336),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        # 读取元数据
        self.read_mat_metadata()

        # 构建Dataset
        self.train_dataset = ImgDataset(self.image_files[self.trainval_loc],
                                        self.labels[self.trainval_loc],
                                        'CUB', self.train_transform, self.root_dir)
        self.test_seen_dataset = ImgDataset(self.image_files[self.test_seen_loc],
                                            self.labels[self.test_seen_loc],
                                            'CUB', self.test_transform, self.root_dir)
        self.test_unseen_dataset = ImgDataset(self.image_files[self.test_unseen_loc],
                                              self.labels[self.test_unseen_loc],
                                              'CUB', self.test_transform, self.root_dir)

        # 统计数据
        self.ntrain_clip = len(self.train_dataset)
        self.ntrain_class = len(self.seenclasses)
        self.ntest_class = len(self.unseenclasses)
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        self.get_idx_classes()

        # 构建 DataLoader
        self.test_seen_loader = DataLoader(self.test_seen_dataset, batch_size=50, shuffle=False, num_workers=8)
        self.test_unseen_loader = DataLoader(self.test_unseen_dataset, batch_size=50, shuffle=False, num_workers=8)

    def read_mat_metadata(self):
        mat_path = os.path.join(self.data_path, 'data/xlsa17/data', self.dataset, 'res101.mat')
        split_path = os.path.join(self.data_path, 'data/xlsa17/data', self.dataset, 'att_splits.mat')

        # 读取图片路径
        res101 = sio.loadmat(mat_path)
        self.image_files = np.squeeze(res101['image_files'])
        self.labels = res101['labels'].astype(int).squeeze() - 1

        # 读取标签和分割
        att_splits = sio.loadmat(split_path)
        self.trainval_loc = att_splits['trainval_loc'].squeeze() - 1
        self.test_seen_loc = att_splits['test_seen_loc'].squeeze() - 1
        self.test_unseen_loc = att_splits['test_unseen_loc'].squeeze() - 1

        self.seenclasses = torch.from_numpy(np.unique(self.labels[self.trainval_loc])).to(self.device)
        self.unseenclasses = torch.from_numpy(np.unique(self.labels[self.test_unseen_loc])).to(self.device)

        # 读取专家属性
        att = att_splits['att'].T
        self.att = torch.from_numpy(att).float().to(self.device)

        # ============================================================
        # [修改] 直接从 att_splits.mat 中读取 'allclasses_names'
        # ============================================================
        if 'allclasses_names' in att_splits:
            # .mat 文件中的字符串通常被包在 numpy array 里，格式为 [[array(['name'], dtype='<U...')], ...]
            # 需要用 squeeze() 拉平，然后取出第0个元素并转为 str
            self.class_names = [str(c[0]) for c in att_splits['allclasses_names'].squeeze()]
            print(f"Successfully loaded {len(self.class_names)} class names from att_splits.mat")
        else:
            raise ValueError(f"'allclasses_names' key not found in {split_path}!")
        # ============================================================

        # 读取clip语义向量
        clip_att_path = os.path.join(self.data_path, 'data/clip_att', f'{self.dataset}_attribute.pkl')
        if os.path.exists(clip_att_path):
            with open(clip_att_path, 'rb') as f:
                self.clip_att = pickle.load(f)
            if not isinstance(self.clip_att, torch.Tensor):
                self.clip_att = torch.from_numpy(self.clip_att)
            self.clip_att = self.clip_att.float().to(self.device)
        else:
            print(f"ERROR: CLIP attribute file not found at {clip_att_path}")
            raise FileNotFoundError(f"Missing {clip_att_path}")

    def get_idx_classes(self):
        # 均衡采样辅助索引
        self.idxs_list = []
        train_labels = self.labels[self.trainval_loc]
        for i in range(self.ntrain_class):
            label_c = self.seenclasses[i].item()
            idx_c = np.where(train_labels == label_c)[0]
            self.idxs_list.append(idx_c)

    def next_batch(self, batch_size):
        # 保持 Balanced Sampling 逻辑
        if self.is_balance:
            batch_idxs_local = []
            n_samples_class = max(batch_size // self.ntrain_class, 1)
            sampled_classes = np.random.choice(np.arange(self.ntrain_class),
                                               min(self.ntrain_class, batch_size), replace=False)
            for i_c in sampled_classes:
                idxs = self.idxs_list[i_c]
                sampled_idx = np.random.choice(idxs, n_samples_class)
                batch_idxs_local.append(sampled_idx)
            batch_idxs_local = np.concatenate(batch_idxs_local)
        else:
            batch_idxs_local = np.random.choice(self.ntrain_clip, batch_size, replace=False)

        # 实时加载图片
        batch_images = []
        batch_labels = []
        for idx in batch_idxs_local:
            img, label = self.train_dataset[idx]
            batch_images.append(img)
            batch_labels.append(label)

        # 堆叠 Tensor
        batch_images = torch.stack(batch_images).to(self.device)
        batch_labels = torch.tensor(batch_labels).long().to(self.device)
        batch_att = self.att[batch_labels]
        return batch_labels, batch_images, batch_att

class SUNDataLoader():
    def __init__(self, data_path, device, is_scale=False, is_unsupervised_attr=False, is_balance=True):
        self.data_path = data_path
        self.device = device
        # ================= 修改点 1: 数据集名称改名为 SUN =================
        self.dataset = 'SUN'
        # =================================================================

        # 假设 SUN 图片的根目录类似于 ./data/SUN
        self.root_dir = os.path.join(self.data_path, 'data', self.dataset)

        self.is_balance = is_balance
        self.is_unsupervised_attr = is_unsupervised_attr

        print(f"Loading metadata from {self.root_dir}...")

        # 定义 CLIP 预处理 (与 CUB 保持一致)
        self.transform = transforms.Compose([
            transforms.Resize((336, 336), interpolation=Image.BICUBIC),
            transforms.CenterCrop(336),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        # 测试集预处理
        self.test_transform = transforms.Compose([
            transforms.Resize((336, 336), interpolation=Image.BICUBIC),
            transforms.CenterCrop(336),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        # 读取元数据
        self.read_mat_metadata()

        # 构建 Dataset
        self.train_dataset = ImgDataset(self.image_files[self.trainval_loc],
                                        self.labels[self.trainval_loc],
                                        self.dataset, self.transform, self.root_dir)

        self.test_seen_dataset = ImgDataset(self.image_files[self.test_seen_loc],
                                            self.labels[self.test_seen_loc],
                                            self.dataset, self.test_transform, self.root_dir)

        self.test_unseen_dataset = ImgDataset(self.image_files[self.test_unseen_loc],
                                              self.labels[self.test_unseen_loc],
                                              self.dataset, self.test_transform, self.root_dir)

        # 统计数据
        self.ntrain_clip = len(self.train_dataset)
        self.ntrain_class = len(self.seenclasses)
        self.ntest_class = len(self.unseenclasses)
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        self.get_idx_classes()

        # 构建 DataLoader
        self.test_seen_loader = DataLoader(self.test_seen_dataset, batch_size=64, shuffle=False, num_workers=8)
        self.test_unseen_loader = DataLoader(self.test_unseen_dataset, batch_size=64, shuffle=False, num_workers=8)

    def read_mat_metadata(self):
        # 路径指向原始的 .mat 文件
        mat_path = os.path.join(self.data_path, 'data/xlsa17/data', self.dataset, 'res101.mat')
        split_path = os.path.join(self.data_path, 'data/xlsa17/data', self.dataset, 'att_splits.mat')

        # 读取图片路径
        res101 = sio.loadmat(mat_path)
        self.image_files = np.squeeze(res101['image_files'])
        self.labels = res101['labels'].astype(int).squeeze() - 1

        # 读取标签和分割
        att_splits = sio.loadmat(split_path)
        self.trainval_loc = att_splits['trainval_loc'].squeeze() - 1
        self.test_seen_loc = att_splits['test_seen_loc'].squeeze() - 1
        self.test_unseen_loc = att_splits['test_unseen_loc'].squeeze() - 1

        self.seenclasses = torch.from_numpy(np.unique(self.labels[self.trainval_loc])).to(self.device)
        self.unseenclasses = torch.from_numpy(np.unique(self.labels[self.test_unseen_loc])).to(self.device)

        # 读取专家属性 (SUN 有 102 维属性)
        att = att_splits['att'].T
        self.att = torch.from_numpy(att).float().to(self.device)

        # ================= [新增关键修改: 读取类别名称] =================
        # 用于 Residual Fusion 生成文本特征
        if 'allclasses_names' in att_splits:
            # .mat 文件中的字符串通常被包在 numpy array 里，需要转为 str
            self.class_names = [str(c[0]) for c in att_splits['allclasses_names'].squeeze()]
            print(f"Successfully loaded {len(self.class_names)} class names for SUN from att_splits.mat")
        else:
            raise ValueError(f"'allclasses_names' key not found in {split_path}!")
        # ===============================================================

        # 读取 CLIP 语义向量
        clip_att_path = os.path.join(self.data_path, 'data/clip_att', f'{self.dataset}_attribute.pkl')
        if os.path.exists(clip_att_path):
            with open(clip_att_path, 'rb') as f:
                self.clip_att = pickle.load(f)
            if not isinstance(self.clip_att, torch.Tensor):
                self.clip_att = torch.from_numpy(self.clip_att)
            self.clip_att = self.clip_att.float().to(self.device)
        else:
            print(f"ERROR: CLIP attribute file not found at {clip_att_path}")
            raise FileNotFoundError(f"Missing {clip_att_path}")

    def get_idx_classes(self):
        self.idxs_list = []
        train_labels = self.labels[self.trainval_loc]
        for i in range(self.ntrain_class):
            label_c = self.seenclasses[i].item()
            idx_c = np.where(train_labels == label_c)[0]
            self.idxs_list.append(idx_c)

    def next_batch(self, batch_size):
        if self.is_balance:
            batch_idxs_local = []
            n_samples_class = max(batch_size // self.ntrain_class, 1)
            sampled_classes = np.random.choice(np.arange(self.ntrain_class),
                                               min(self.ntrain_class, batch_size), replace=False)
            for i_c in sampled_classes:
                idxs = self.idxs_list[i_c]
                sampled_idx = np.random.choice(idxs, n_samples_class)
                batch_idxs_local.append(sampled_idx)
            batch_idxs_local = np.concatenate(batch_idxs_local)
        else:
            batch_idxs_local = np.random.choice(self.ntrain_clip, batch_size, replace=False)

        batch_images = []
        batch_labels = []
        for idx in batch_idxs_local:
            img, label = self.train_dataset[idx]
            batch_images.append(img)
            batch_labels.append(label)

        batch_images = torch.stack(batch_images).to(self.device)
        batch_labels = torch.tensor(batch_labels).long().to(self.device)
        batch_att = self.att[batch_labels]
        return batch_labels, batch_images, batch_att

class AWA2DataLoader():
    def __init__(self, data_path, device, is_scale=False, is_unsupervised_attr=False, is_balance=True):
        self.data_path = data_path
        self.device = device
        self.dataset = 'AWA2'

        # 假设 AWA2 图片根目录在 ./data/AWA2
        self.root_dir = os.path.join(self.data_path, 'data', self.dataset)

        self.is_balance = is_balance
        self.is_unsupervised_attr = is_unsupervised_attr

        print(f"Loading metadata from {self.root_dir}...")

        # 定义 CLIP 预处理
        self.transform = transforms.Compose([
            transforms.Resize((336, 336), interpolation=Image.BICUBIC),
            transforms.CenterCrop(336),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        # 测试集预处理 (保持一致)
        self.test_transform = transforms.Compose([
            transforms.Resize((336, 336), interpolation=Image.BICUBIC),
            transforms.CenterCrop(336),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        # 读取元数据
        self.read_mat_metadata()

        # 构建 Dataset
        self.train_dataset = ImgDataset(self.image_files[self.trainval_loc],
                                        self.labels[self.trainval_loc],
                                        self.dataset, self.transform, self.root_dir)

        self.test_seen_dataset = ImgDataset(self.image_files[self.test_seen_loc],
                                            self.labels[self.test_seen_loc],
                                            self.dataset, self.test_transform, self.root_dir)

        self.test_unseen_dataset = ImgDataset(self.image_files[self.test_unseen_loc],
                                              self.labels[self.test_unseen_loc],
                                              self.dataset, self.test_transform, self.root_dir)

        # 统计数据
        self.ntrain_clip = len(self.train_dataset)
        self.ntrain_class = len(self.seenclasses)
        self.ntest_class = len(self.unseenclasses)
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        self.get_idx_classes()

        # 构建 DataLoader (Batch Size 建议开大一点，如 128，因为 AWA2 图片多)
        self.test_seen_loader = DataLoader(self.test_seen_dataset, batch_size=128, shuffle=False, num_workers=4)
        self.test_unseen_loader = DataLoader(self.test_unseen_dataset, batch_size=128, shuffle=False, num_workers=4)

    def read_mat_metadata(self):
        # 自动拼接路径: data/xlsa17/data/AWA2/res101.mat
        mat_path = os.path.join(self.data_path, 'data/xlsa17/data', self.dataset, 'res101.mat')
        split_path = os.path.join(self.data_path, 'data/xlsa17/data', self.dataset, 'att_splits.mat')

        # 读取图片路径
        res101 = sio.loadmat(mat_path)
        self.image_files = np.squeeze(res101['image_files'])
        self.labels = res101['labels'].astype(int).squeeze() - 1

        # 读取标签和分割
        att_splits = sio.loadmat(split_path)
        self.trainval_loc = att_splits['trainval_loc'].squeeze() - 1
        self.test_seen_loc = att_splits['test_seen_loc'].squeeze() - 1
        self.test_unseen_loc = att_splits['test_unseen_loc'].squeeze() - 1

        self.seenclasses = torch.from_numpy(np.unique(self.labels[self.trainval_loc])).to(self.device)
        self.unseenclasses = torch.from_numpy(np.unique(self.labels[self.test_unseen_loc])).to(self.device)

        # 读取专家属性 (AWA2 有 85 维)
        att = att_splits['att'].T
        self.att = torch.from_numpy(att).float().to(self.device)

        # ================= [新增关键修改: 读取类别名称] =================
        if 'allclasses_names' in att_splits:
            # 读取所有类别名称
            self.class_names = [str(c[0]) for c in att_splits['allclasses_names'].squeeze()]
            print(f"Successfully loaded {len(self.class_names)} class names for AWA2.")
        else:
            raise ValueError(f"'allclasses_names' key not found in {split_path}!")
        # ===============================================================

        # 读取 CLIP 语义向量
        clip_att_path = os.path.join(self.data_path, 'data/clip_att', f'{self.dataset}_attribute.pkl')
        if os.path.exists(clip_att_path):
            with open(clip_att_path, 'rb') as f:
                self.clip_att = pickle.load(f)
            if not isinstance(self.clip_att, torch.Tensor):
                self.clip_att = torch.from_numpy(self.clip_att)
            self.clip_att = self.clip_att.float().to(self.device)
        else:
            print(f"ERROR: CLIP attribute file not found at {clip_att_path}")
            raise FileNotFoundError(f"Missing {clip_att_path}")

    def get_idx_classes(self):
        self.idxs_list = []
        train_labels = self.labels[self.trainval_loc]
        for i in range(self.ntrain_class):
            label_c = self.seenclasses[i].item()
            idx_c = np.where(train_labels == label_c)[0]
            self.idxs_list.append(idx_c)

    def next_batch(self, batch_size):
        if self.is_balance:
            batch_idxs_local = []
            n_samples_class = max(batch_size // self.ntrain_class, 1)
            sampled_classes = np.random.choice(np.arange(self.ntrain_class),
                                               min(self.ntrain_class, batch_size), replace=False)
            for i_c in sampled_classes:
                idxs = self.idxs_list[i_c]
                sampled_idx = np.random.choice(idxs, n_samples_class)
                batch_idxs_local.append(sampled_idx)
            batch_idxs_local = np.concatenate(batch_idxs_local)
        else:
            batch_idxs_local = np.random.choice(self.ntrain_clip, batch_size, replace=False)

        batch_images = []
        batch_labels = []
        for idx in batch_idxs_local:
            img, label = self.train_dataset[idx]
            batch_images.append(img)
            batch_labels.append(label)

        batch_images = torch.stack(batch_images).to(self.device)
        batch_labels = torch.tensor(batch_labels).long().to(self.device)
        batch_att = self.att[batch_labels]
        return batch_labels, batch_images, batch_att