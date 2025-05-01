# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# fixed seed for experiment
np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        # MLP
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
        # CNN
        # train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, 28, 28)

with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)

from scipy.ndimage import rotate, shift
import numpy as np

def augment_mnist(images, labels):
    augmented_images = []
    augmented_labels = []

    for img, label in zip(images, labels):
        img = img.reshape(28, 28)

        # 随机旋转 [-10°, +10°]
        rotated = rotate(img, angle=np.random.uniform(-10, 10), reshape=False, mode='constant')

        # 随机平移 [-2, +2] 像素
        shifted = shift(img, shift=(np.random.uniform(-2, 2), np.random.uniform(-2, 2)), mode='constant')

        # 添加增强图像（展平）
        augmented_images.append(rotated.flatten())
        augmented_images.append(shifted.flatten())
        augmented_labels.extend([label, label])

    # 拼接原始与增强数据
    new_images = np.concatenate([images, np.array(augmented_images)], axis=0)
    new_labels = np.concatenate([labels, np.array(augmented_labels)], axis=0)

    return new_images, new_labels


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
# 增强原始数据（可多次）

train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

# train_imgs, train_labs = augment_mnist(train_imgs, train_labs)

# 一个隐藏层
# linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])
# 两个隐藏层
# linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 256, 128, 10], 'ReLU', [1e-4, 1e-4, 1e-4])
linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 256, 128, 10], 'ReLU', [0.0, 0.0, 0.0])
# CNN模型
# cnn_model = nn.models.Model_CNN()

# SGD 优化
# optimizer = nn.optimizer.SGD(init_lr=0.06, model=linear_model)
# 动量优化
optimizer = nn.optimizer.MomentGD(init_lr=0.06, model=linear_model, mu = 0.9)

scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

# 启用 Dropout
# for layer in cnn_model.layers:
#     if isinstance(layer, nn.op.Dropout):
#         layer.training = True

# 训练MLP
runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=100, save_dir=r'./best_models')
# 训练CNN
# runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=2, log_iters=10, save_dir=r'./best_models')

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.show()