import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

# from models.vgg import VGG_A
# from models.vgg import VGG_A_Light
# from models.vgg import VGG_A_Dropout
from models.vgg import ResNet_CIFAR10
from models.vgg import VGG_A_BatchNorm # you need to implement this network
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
device_id = [0,1,2,3]
num_workers = 4
batch_size = 128

# add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

# This function is used to calculate the accuracy of model classification
def get_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, preds = outputs.max(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total
    model.train()
    return acc

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def visualize_filters(model, save_path):
    first_conv = None
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            first_conv = layer
            break
    if first_conv is not None:
        weight = first_conv.weight.data.clone().cpu()
        n_kernels = min(weight.shape[0], 16)
        plt.figure(figsize=(12, 6))
        for i in range(n_kernels):
            kernel = weight[i]
            kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min()) 
            kernel = kernel.permute(1, 2, 0).numpy()
            plt.subplot(2, 8, i+1)
            plt.imshow(kernel)
            plt.axis('off')
        plt.suptitle("Visualization of the first layer of convolutional kernels")
        plt.savefig(save_path)
        plt.close()

def visualize_feature_maps(model, input_tensor, save_path='reports/figures/feature_maps.png'):
    model.eval()
    with torch.no_grad():
        x = input_tensor.unsqueeze(0).to(device)
        for layer in model.features:
            x = layer(x)
            if isinstance(layer, nn.ReLU):  
                break
        fmap = x.squeeze(0).cpu()
        num_maps = min(16, fmap.shape[0])
        plt.figure(figsize=(12, 6))
        for i in range(num_maps):
            plt.subplot(2, 8, i+1)
            plt.imshow(fmap[i], cmap='viridis')
            plt.axis('off')
        plt.suptitle('The first layer outputs feature maps')
        plt.savefig(save_path)
        plt.close()

# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = []
    train_accuracy_curve = []
    val_accuracy_curve = []
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    grads = []

    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_sum = 0
        grad = []

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()

        losses_list.append(loss_sum / batches_n)
        grads.append(grad)

        train_acc = get_accuracy(model, train_loader, device)
        val_acc = get_accuracy(model, val_loader, device)
        learning_curve.append(loss_sum / batches_n)
        train_accuracy_curve.append(train_acc)
        val_accuracy_curve.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs_n} - Train Loss: {loss_sum / batches_n:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > max_val_accuracy and best_model_path:
            max_val_accuracy = val_acc
            max_val_accuracy_epoch = epoch+1
            torch.save(model.state_dict(), best_model_path)
            print("Best model saved at epoch", max_val_accuracy_epoch)

    print("Training finished. Best val acc:", max_val_accuracy)
    return losses_list, grads, train_accuracy_curve, val_accuracy_curve, learning_curve

# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()
def plot_loss_landscape(min_curve, max_curve):
    plt.figure(figsize=(8,4))
    plt.fill_between(range(len(min_curve)), min_curve, max_curve, color='skyblue', alpha=0.5, label='Loss Landscape')
    plt.plot(min_curve, label='Min Loss')
    plt.plot(max_curve, label='Max Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Landscape')
    plt.savefig(os.path.join(figures_path, 'loss_landscape.png'))
    plt.close()

if __name__ == "__main__":
    loss_save_path = ''
    grad_save_path = ''

    os.makedirs(figures_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(loss_save_path if loss_save_path else '.', exist_ok=True)
    os.makedirs(grad_save_path if grad_save_path else '.', exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    set_random_seeds(seed_value=2020, device=device)

    train_loader = get_cifar_loader(train=True)
    val_loader = get_cifar_loader(train=False)

    # 可视化一张训练样本
    for X, y in train_loader:
        img = X[0].cpu().numpy().transpose(1, 2, 0)
        plt.imshow(img * 0.5 + 0.5)
        plt.title(f"Label: {y[0].item()}")
        plt.savefig(os.path.join(figures_path, 'sample_batch.png'))
        print("Saved a sample figure.")
        break

    epo = 20
    model = ResNet_CIFAR10()
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_model_path = os.path.join(models_path, "vgg_a_best.pth")
    loss, grads, train_acc_curve, val_acc_curve, learning_curve = train(
        model, optimizer, criterion,
        train_loader, val_loader,
        epochs_n=epo, best_model_path=best_model_path
    )

    np.savetxt(os.path.join(loss_save_path, 'loss.txt'), loss, fmt='%s', delimiter=' ')
    np.savetxt(os.path.join(grad_save_path, 'grads.txt'), grads, fmt='%s', delimiter=' ')

    # loss landscape 
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epo+1), learning_curve[:epo], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Landscape')
    plt.savefig(os.path.join(figures_path, 'loss_landscape.png'))
    plt.close()
    print("Loss landscape figure saved!")

    # acc/loss 变化曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epo+1), learning_curve[:epo], label='train loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epo+1), train_acc_curve[:epo], label='train accuracy')
    plt.plot(range(1, epo+1), val_acc_curve[:epo], label='validation accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.suptitle("Training and validation accuracy/loss change curves")
    plt.savefig(os.path.join(figures_path, 'acc_loss_curve.png'))
    plt.close()
    print("Saved acc_loss_curve.png")

    # 可视化卷积核
    visualize_filters(model, os.path.join(figures_path, 'conv1_kernels.png'))
    print("Saved conv filters visualization.")

    # 可视化特征图（用验证集一张图）
    for X_val, _ in val_loader:
        visualize_feature_maps(model, X_val[0], os.path.join(figures_path, 'feature_maps.png'))
        print("Saved feature maps visualization.")
        break
