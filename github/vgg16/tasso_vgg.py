import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training, extract_face
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torchvision import datasets, transforms, models
import numpy as np
import os
import torch.optim as optim
import torch.utils.data as data
import pandas as pd

data_dir = '.../vgg16/data'

batch_size = 1
workers = 4 if os.name == 'nt' else 8

#escolher dispositivo
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

#para mudar tamanho da imagem
mtcnn = MTCNN(
    image_size=160,
    margin=14,
    device=device,
    selection_method='center_weighted_size'
)
orig_img_ds = datasets.ImageFolder(data_dir, transform=None)
orig_img_ds.samples = [
    (p, p)
    for p, _ in orig_img_ds.samples
]

loader = DataLoader(
    orig_img_ds,
    num_workers=workers,
    batch_size=batch_size,
    collate_fn=training.collate_pil
)
crop_paths = []
box_probs = []

for i, (x, b_paths) in enumerate(loader):
    crops = [p for p in b_paths]
    mtcnn(x, save_path=crops)
    crop_paths.extend(crops)
    print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')
    
del mtcnn
torch.cuda.empty_cache()

trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])

dataset = datasets.ImageFolder(data_dir, transform=trans)

embed_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SequentialSampler(dataset)
)
x = []
y = []
with torch.no_grad():
    for xb, yb in embed_loader:
        x.extend(xb.numpy())
        y.extend(yb.numpy())
x = torch.tensor(x)
y = torch.tensor(y)

class_names = dataset.classes

print('Label 0 corresponde à classe: {}'.format(class_names[0]))
print('Label 1 corresponde à classe: {}'.format(class_names[1]))

##////////////////////////////////////////////////////////////////////////////////////////////////////////////
def save_checkpoint(state, filename='best_model.pth.tar'):
    torch.save(state, filename)
    
def train_model(net, X_train, y_train, X_test, y_test, criterion, optimizer, num_epochs, save_adr):
    accuracy_list = []
    loss_list = []
    accuracy_train_list = []
    accuracy_train_list.append(0)
    loss_train_list = []
    loss_train_list.append(0)
    net.to(device)
    torch.backends.cudnn.benchmark = True
    best_val_loss = 1000
    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  
                xuse = X_train
                yuse = y_train
            else:
                net.eval()  
                xuse = X_test
                yuse = y_test
            epoch_loss = 0.0
            epoch_corrects = 0
            all_preds = []
            all_labels = []
            xuse = np.expand_dims(xuse,1)
            yuse = np.expand_dims(yuse,1)
            if (epoch == 0) and (phase == 'train'):
                continue
            xandy = []
            for i in range(len(xuse)):
                xandy.append((xuse[i],yuse[i]))
            for (inputs, labels) in xandy:

                inputs = torch.tensor(inputs)
                labels = torch.tensor(labels)
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)

                    loss = criterion(outputs, labels)  
                    _, preds = torch.max(outputs, 1)  

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)
                    
                    all_preds.append(preds)
                    all_labels.append(labels)

            epoch_loss = epoch_loss / len(xuse)
            epoch_acc = epoch_corrects.double() / len(xuse)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if phase == 'train':
                accuracy_train_list.append(epoch_acc.item())
                loss_train_list.append(epoch_loss)
            elif phase == 'val':
                accuracy_list.append(epoch_acc.item())
                loss_list.append(epoch_loss)
                #Verifica se houve melhora
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_epoch = epoch
                    
                    all_preds = torch.cat(all_preds)
                    all_labels = torch.cat(all_labels)
                    

                    
                    net_statedict = net.state_dict()
                    optimizer_state = optimizer.state_dict()
                    
        torch.cuda.empty_cache()        
    save_checkpoint({'epoch': best_epoch+1, 'state_dict': net_statedict, 'optimizer': optimizer_state, 'val_loss': best_val_loss}, filename=save_adr)      
               
    return best_epoch, accuracy_list, loss_list, accuracy_train_list, loss_train_list, 

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_accuracy_list = []
all_loss_list = []
best_epochs_list = []

i = 1
cross_saves = ['best_model_cross_1.pth.tar','best_model_cross_2.pth.tar','best_model_cross_3.pth.tar','best_model_cross_4.pth.tar','best_model_cross_5.pth.tar']

#Treinamento com 5-Fold da rede, descongelando as FullyConnected e as 3 Últimas camadas convolutivas
for train_index, val_index in kf.split(x):
    df = pd.read_csv('.../vgg16/vgg_log.csv')

    model = models.vgg16(pretrained=True, progress=True).to(device)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=2)
    model.train()

    criterion = nn.CrossEntropyLoss()

    params_to_update_1 = []
    params_to_update_2 = []
    params_to_update_3 = []

    update_param_names_1 = ["features.24","features.28", "features.26"]
    update_param_names_2 = ["classifier.0.weight",
                            "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
    update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]

    for name, param in model.named_parameters():
        if update_param_names_1[0] in name:
            param.requires_grad = True
            params_to_update_1.append(param)

        elif name in update_param_names_2:
            param.requires_grad = True
            params_to_update_2.append(param)

        elif name in update_param_names_3:
            param.requires_grad = True
            params_to_update_3.append(param)

        else:
            param.requires_grad = False

    optimizer = optim.SGD([
        {'params': params_to_update_1, 'lr': 0.0001},
        {'params': params_to_update_2, 'lr': 0.0001},
        {'params': params_to_update_3, 'lr': 0.0001}
    ], momentum=0.9)

    
    X_train, X_val = x[train_index], x[val_index]
    y_train, y_val = y[train_index], y[val_index]

    best_epoch, accuracy_list, loss_list, accuracy_train_list, loss_train_list= train_model(model,X_train, y_train, X_val, y_val, criterion, optimizer, num_epochs=150, save_adr = cross_saves[i-1])
    
    dataframe = []
    #Salva os resultados em um arquivo csv
    for epoca in range(len(accuracy_train_list)):
        dictt = {'Validação': i, 'Melhor Época': best_epoch, 'Época': epoca, 'Acurácia de Treino': accuracy_train_list[epoca], 'Loss de Treino':loss_train_list[epoca], 'Acurácia de Teste':accuracy_list[epoca], 'Loss de teste':loss_list[epoca] }
        dataframe.append(dictt)
    dataframe = pd.DataFrame(dataframe)
    df = pd.concat([df, dataframe], ignore_index=True)
    df.to_csv('.../vgg_16/vgg_log.csv', index=False)


    valor_max = np.max(accuracy_list)
    accuracy_max_idx = np.where(accuracy_list==valor_max)[0]
    loss_list_acc = [(i,loss_list[i]) for i in accuracy_max_idx]
    min_loss_tuple = min(loss_list_acc, key=lambda x: x[1])
    loss_max_idx = min_loss_tuple[0]
    idx =loss_max_idx
    best_epochs_list.append(idx+1)
    all_accuracy_list.append(accuracy_list[idx])
    all_loss_list.append(loss_list[idx])
    i+=1
    
# Média das acurácias e perdas
mean_accuracy = np.mean(all_accuracy_list, axis=0)
mean_loss = np.mean(all_loss_list, axis=0)
print("Melhor Época de cada validação:", best_epochs_list)
print("Lista das Acurácias:", all_accuracy_list)
print("Lista das Perdas:", all_loss_list)
print("Média das Acurácias:", mean_accuracy)
print("Média das Perdas:", mean_loss)

