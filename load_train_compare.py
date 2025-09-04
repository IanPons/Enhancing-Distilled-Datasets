import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from utils import get_dataset, TensorDataset, get_default_convnet_setting
import random


class ConvNet(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        super(ConvNet, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def embed(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == 'swish':
            return Swish()
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat

def set_seed(seed=21):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Para múltiplas GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Pode impactar desempenho, mas garante reprodutibilidade


def combine_datasets(dataset1, dataset2, alpha):

    images1, labels1 = dataset1.images.cpu().numpy(), dataset1.labels.cpu().numpy()
    images2, labels2 = dataset2.images.cpu().numpy(), dataset2.labels.cpu().numpy()
    
    label_to_indices = {label: np.where(labels2 == label)[0] for label in np.unique(labels2)}
    
    random_indices = np.array([np.random.choice(label_to_indices[label]) for label in labels1])
    
    selected_images2 = images2[random_indices]
    
    combined_images = (1 - alpha) * images1 + alpha * selected_images2
    
    combined_images_tensor = torch.tensor(combined_images).to('cuda')
    combined_labels_tensor = torch.tensor(labels1).to('cuda')  # As labels não mudam
    return TensorDataset(combined_images_tensor, combined_labels_tensor)

# Função de treinamento
def epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / len(train_loader), 100. * correct / total

# Função de validação
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return running_loss / len(val_loader), 100. * correct / total

def train_model(epochs, channel, num_classes, train_loader, testloader, device, evaluate_every):
    set_seed()

    val_acc = None

    learning_rate = 0.001
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    model = ConvNet(channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # Treinamento
    for ep in range(epochs):
        train_loss, train_acc = epoch(model, train_loader, criterion, optimizer, device)

        # print(f"Epoch [{epoch+1}/{epochs}]")
        # print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

        if (ep+ 1) % evaluate_every == 0:
            val_loss, val_acc = validate(model, testloader, criterion, device)
            print(f"Epoch [{ep+1}/{epochs}] --- Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

    return model, val_acc

def train_model_combined(epochs, channel, num_classes, dst_distilled, dst_original, testloader, device, evaluate_every):
    set_seed()

    val_acc = None

    learning_rate = 0.001
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    model = ConvNet(channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Treinamento
    for ep in range(epochs):
        alpha = np.random.uniform(0, 1)

        dst_combined = combine_datasets(dst_distilled, dst_original, alpha)
        trainloader = torch.utils.data.DataLoader(dst_combined, batch_size=256, shuffle=True, num_workers=0)

        train_loss, train_acc = epoch(model, trainloader, criterion, optimizer, device)

        # print(f"Epoch [{epoch+1}/{epochs}]")
        # print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

        if (ep+ 1) % evaluate_every == 0:
            val_loss, val_acc = validate(model, testloader, criterion, device)
            print(f"Epoch [{ep+1}/{epochs}] --- Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

    return model, val_acc


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--model', type=str, default='ConvNet', help='architecture')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--original_epochs', type=int, default=100)
    parser.add_argument('--distilled_epochs', type=int, default=300)
    parser.add_argument('--distilled_path', type=str)

    args = parser.parse_args()

    distilled_epochs = args.distilled_epochs
    original_epochs = args.original_epochs

    # Configurações iniciais
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    channel, im_size, num_classes, class_names, mean, std, dst_train_original, dst_test_original, testloader = get_dataset(args.dataset, args.data_path)
    trainloader_original = DataLoader(dst_train_original, batch_size=256, shuffle=True)

    images_all = [torch.unsqueeze(dst_train_original[i][0], dim=0) for i in range(len(dst_train_original))]
    labels_all = [dst_train_original[i][1] for i in range(len(dst_train_original))]
    images_all = torch.cat(images_all, dim=0).to(device)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=device)

    # load distilled data
    data = torch.load(f'./result/{args.distilled_path}.pt')['data'][0]
    images_distilled, labels_distilled = data[0], data[1]  
    dst_distilled = TensorDataset(images_distilled, labels_distilled)
     
    trainloader_distilled = DataLoader(dst_distilled, batch_size=256, shuffle=True)

    # print("Original training:\n", flush=True)
    # model_original, original_acc = train_model(original_epochs, channel, num_classes, trainloader_original, testloader, device, 5)
    # print("Original acc: ", original_acc,  flush=True)
    # torch.save(model_original.state_dict(), f'./models/CIFAR100_original')

    print("Combined training:", flush=True)
    model_combined, combined_acc = train_model_combined(distilled_epochs, channel, num_classes, dst_distilled, TensorDataset(images_all, labels_all), testloader, device, 50)
    print("Combined acc: ", combined_acc,  flush=True)
    torch.save(model_combined.state_dict(), f'./models/Combined_{args.distilled_path}')
    
    print("Distilled training:\n", flush=True)
    model_distilled, distilled_acc = train_model(distilled_epochs, channel, num_classes, trainloader_distilled, testloader, device, 50)
    print("Distilled acc: ", distilled_acc,  flush=True)
    torch.save(model_distilled.state_dict(), f'./models/{args.distilled_path}')

