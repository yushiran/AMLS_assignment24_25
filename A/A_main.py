import os,datetime
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO, Evaluator
from copy import deepcopy
from tqdm import trange
import matplotlib.pyplot as plt
from linformer import Linformer
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import ResNet18
from model import ViT

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def A_main_function(BATCH_SIZE:int, train_or_test:str='train',train_model:str='ResNet18',data_flag:str = 'breastmnist',model_path:str=None):
    gpu_ids = [0]  # Assuming you want to use the first GPU
    device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids and torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # load the data
    download = True
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    # preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # load the data
    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    val_dataset = DataClass(split='val', transform=data_transform, download=download)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)

    frames = train_dataset.montage(length=8, save_folder=f"{CURRENT_DIR}/docs")

    # encapsulate data into dataloader form
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

    train_evaluator = medmnist.Evaluator(data_flag, 'train')
    val_evaluator = medmnist.Evaluator(data_flag, 'val')
    test_evaluator = medmnist.Evaluator(data_flag, 'test')

    # train the model
    if train_or_test == 'train':
        return train(train_model,data_flag=data_flag,device=device,n_channels=n_channels,n_classes=n_classes,task=task,
                    train_loader=train_loader,val_loader=val_loader,test_loader=test_loader,
                    train_evaluator=test_evaluator,val_evaluator=val_evaluator,test_evaluator=test_evaluator)
    # test the model
    elif train_or_test == 'test':
        return inference(model_path=model_path,data_flag=data_flag,device=device,
                        n_channels=n_channels,n_classes=n_classes,task=task,
                        train_loader=train_loader,val_loader=val_loader,test_loader=test_loader,
                        train_evaluator=test_evaluator,val_evaluator=val_evaluator,test_evaluator=test_evaluator)

    return 0

def train(train_model:str,data_flag:str,device,
          n_channels,n_classes,
          task,train_loader,val_loader,test_loader,
          train_evaluator,val_evaluator,test_evaluator):
    NUM_EPOCHS = 100
    gamma=0.1
    milestones = [0.5 * NUM_EPOCHS, 0.75 * NUM_EPOCHS]

    current_time = datetime.datetime.now()
    print(f"Current date and time: {current_time}")
    output_root = f'{CURRENT_DIR}/A_model/{train_model}_{current_time}'
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    if train_model == 'ResNet18':
        lr = 0.001
        model = ResNet18(in_channels=n_channels, num_classes=n_classes)
    elif train_model == 'ViT':
        lr = 3e-5
        feature_dim = 32
        efficient_transformer = Linformer(
                            dim=feature_dim,
                            seq_len=49+1,  # 7x7 patches + 1 cls-token
                            depth=4,
                            heads=4,
                            k=int(0.5*feature_dim)
                        )
        model = ViT(
                    dim=feature_dim,
                    image_size=28,
                    patch_size=4,
                    num_classes=n_classes,
                    transformer=efficient_transformer,
                    channels=n_channels,
                )
    else:
        raise NotImplementedError
    
    model = model.to(device)

    # define loss function and optimizer
    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    best_auc = 0
    best_acc = 0
    best_epoch = 0
    best_model = deepcopy(model)

    global iteration
    iteration = 0

    train_loss = []
    val_loss = []
    progress_bar = trange(NUM_EPOCHS, desc="Training", unit="epoch")
    for epoch in progress_bar:           
        total_loss = []
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                loss = criterion(outputs, targets)
            else:
                targets = torch.squeeze(targets, 1).long().to(device)
                loss = criterion(outputs, targets)

            total_loss.append(loss.item())
            iteration += 1
            loss.backward()
            optimizer.step()
        epoch_loss = sum(total_loss)/len(total_loss) 
        train_loss.append(epoch_loss)
        
        val_metrics = test(model, val_evaluator, val_loader, task, criterion, device)
        test_metrics = test(model, test_evaluator, test_loader, task, criterion, device)
        val_loss.append(val_metrics[0])
        
        scheduler.step()
            
        cur_auc = val_metrics[1]
        cur_acc = val_metrics[2]
        if cur_auc > best_auc:
            best_epoch = epoch
            best_auc = cur_auc
            best_acc = cur_acc
            best_model = deepcopy(model)
        
        progress_bar.set_postfix(
            cur_best_auc=f"{best_auc:.4f}", 
            cur_best_acc=f"{best_acc:.4f}", 
            cur_best_epoch=best_epoch
        )

    state = {
        'net': best_model.state_dict(),
    }
    path = os.path.join(output_root, 'best_model.pth')
    torch.save(state, path)

    # Save train and validation loss
    loss_data = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_auc': best_auc,
        'best_epoch': best_epoch,
        'best_acc': best_acc
    }
    loss_path = os.path.join(output_root, 'loss_data.txt')
    with open(loss_path, 'w') as f:
        for key, value in loss_data.items():
            f.write(f"{key}: {value}\n")

    # Plot train and validation loss
    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(output_root, 'loss_plot.png')
    plt.savefig(plot_path)
    plt.close()
    
    return 0 

def test(model, evaluator, data_loader, task, criterion, device):

    model.eval()
    
    total_loss = []
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))
            
            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                loss = criterion(outputs, targets)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = torch.squeeze(targets, 1).long().to(device)
                loss = criterion(outputs, targets)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            total_loss.append(loss.item())
            y_score = torch.cat((y_score, outputs), 0)

        y_score = y_score.detach().cpu().numpy()
        auc, acc = evaluator.evaluate(y_score)
        
        test_loss = sum(total_loss) / len(total_loss)

        return [test_loss, auc, acc]
    
    
def inference(model_path, data_flag, device,
            n_channels,n_classes,
            task,train_loader,val_loader,test_loader,
            train_evaluator,val_evaluator,test_evaluator):

    model_name = model_path.split('_')[0]
    print(f"Model name: {model_name}")
    if model_name == 'ResNet18':
        model =  ResNet18(in_channels=n_channels, num_classes=n_classes)
    elif model_name == 'ViT':
        feature_dim = 32
        efficient_transformer = Linformer(
                            dim=feature_dim,
                            seq_len=49+1,  # 7x7 patches + 1 cls-token
                            depth=4,
                            heads=4,
                            k=int(0.5*feature_dim)
                        )
        model = ViT(
                    dim=feature_dim,
                    image_size=28,
                    patch_size=4,
                    num_classes=n_classes,
                    transformer=efficient_transformer,
                    channels=n_channels,
                )
    else:
        raise NotImplementedError
    model = model.to(device)
    best_model_path = f'{CURRENT_DIR}/A_model/{model_path}/best_model.pth'
    model.load_state_dict(torch.load(best_model_path, map_location=device)['net'], strict=True)

    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    test_metrics = test(model, test_evaluator, test_loader, task, criterion, device)

    output_str = f"Task A, {model_name} model test  auc: {test_metrics[1]}  acc: {test_metrics[2]}"
    print(output_str+'\n')

    # Save test metrics to a text file
    output_root = f'{CURRENT_DIR}/A_model/{model_path}'
    metrics_path = os.path.join(output_root, 'test_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Test AUC: {test_metrics[1]:.5f}\n")
        f.write(f"Test Accuracy: {test_metrics[2]:.5f}\n")

    return output_str


if __name__ == '__main__':
    train_or_test = 'test'
    # train_or_test = 'train'
    # train_model = 'ResNet18'
    train_model = 'ViT'
    BATCH_SIZE = 8
    data_flag = 'breastmnist'
    # model_path = 'ResNet18_2024-12-24 19:40:03.239511'
    model_path = 'ViT_2024-12-25 01:13:58.738999'
 
    A_main_function(BATCH_SIZE=BATCH_SIZE,
                    train_or_test=train_or_test, 
                    train_model=train_model, 
                    data_flag=data_flag,
                    model_path=model_path)