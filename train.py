from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn.functional as F

def train_model(options, model, optimizer, loss_function, train_loader, test_loader):
    scaler = GradScaler()
    loss_history, acc_history, test_history = [], [], []
    params = options['train']
    for epoch in range(params['epochs']):
        model.train()
        loss_val, acc_train, test_acc = 0, 0, 0
        for sample in (pbar := tqdm(train_loader)):
          img, label = sample[0], sample[1]
          img = img.to(options['device'])
          label = label.to(options['device'])
          label = F.one_hot(label, 29).float()
          optimizer.zero_grad()
          with autocast(params['use_amp']):
            pred = model(img)
            loss = loss_function(pred, label)

          scaler.scale(loss).backward()
          loss_item = loss.item()
          loss_val += loss_item

          scaler.step(optimizer)
          scaler.update()

          acc_current = accuracy(pred.cpu().float(), label.cpu().float())
          acc_train += acc_current

          pbar.set_description(f'epoch: {epoch}\tloss: {loss_item:.5f}\taccuracy: {acc_current:.3f}')
        
        loss_history.append(loss_val/len(train_loader))
        acc_history.append(acc_train/len(train_loader))
        print(f'loss: {loss_val/len(train_loader)}')
        print(f'train: {acc_train/len(train_loader)}')
        
        if params['validate']:
            model.eval()
            for sample in test_loader:
                img, label = sample[0], sample[1]
                img = img.to(options['device'])
                label = label.to(options['device'])
                label = F.one_hot(label, 2).float()
                pred = model(img)
                acc_current = accuracy(pred.cpu().float(), label.cpu().float())
                test_acc += acc_current
                test_history.append(test_acc/len(test_loader))
                print(f'test: {test_acc/len(test_loader)}')
        
    return loss_history, acc_history, test_history


def accuracy(pred, label):
    answer = F.softmax(pred.detach()).numpy().argmax(1) == label.numpy().argmax(1)
    return answer.mean()

def make_optimizer(options, model_params):
    params = options['optimizer']
    if params['name'] == 'adam':
        optimizer = torch.optim.Adam(model_params, lr=params['lr'], betas=(params['beta1'], params['beta2']))
    else:
        raise NotImplementedError(f'optimizer {params["optimizer"]} is not implemented')
    return optimizer

