from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn.functional as F


class Trainer():
    def __init__(self, model, optimizer, loss_function, train_loader, test_loader):
        self.model = model
        self.optim = optimizer
        self.loss_function = loss_function
        self.train_loader = train_loader
        self.test_loader = test_loader
    
    def train_model(self, options):
        scaler = GradScaler()
        self.loss_history, self.acc_history, self.val_history = [], [], []
        for epoch in range(options['epochs']):
            train_loss, train_acc = self.train_one_epoch(options, scaler, epoch)
            len_loader = len(self.train_loader)
            loss = train_loss/len_loader
            acc = train_acc/len_loader
            self.loss_history.append(loss)
            self.acc_history.append(acc)
            print(f'loss: {loss}')
            print(f'accuracy: {acc}')
            
            if options['validate']:
                val_acc = self.validate(options)
                len_loader = len(self.test_loader)
                self.val_history.append(val_acc/len_loader)
                print(f'test: {val_acc/len_loader}')
            
    
    def train_one_epoch(self, options, scaler, epoch):
        self.model.train()
        train_loss, train_acc = 0, 0
        for sample in (pbar := tqdm(self.train_loader)):
            img, label = sample[0], sample[1]
            img = img.to(options['device'])
            label = label.to(options['device'])
            label = F.one_hot(label, options['nums_classes']).float()
            self.optim.zero_grad()
            with autocast(options['use_amp']):
                pred = self.model(img)
                loss = self.loss_function(pred, label)
            scaler.scale(loss).backward()
            loss_item = loss.item()
            train_loss += loss_item
            scaler.step(self.optim)
            scaler.update()
            # TODO
            if options['accuracy']:
                acc_current = self.accuracy(pred.cpu().float(), label.cpu().float())
                train_acc += acc_current
            pbar.set_description(f'epoch: {epoch}\tloss: {loss_item:.5f}\taccuracy: {acc_current:.3f}')
        return train_loss, train_acc
    
    def validate(self, options):
        self.model.eval()
        val_acc = 0
        for sample in (pbar := tqdm(self.train_loader)):
            img, label = sample[0], sample[1]
            img = img.to(options['device'])
            label = F.one_hot(label, options['nums_classes']).float()
            with torch.no_grad():
                pred = self.model(img)
            acc_current = self.accuracy(pred.cpu().float(), label.float())
            val_acc += acc_current
        return val_acc
    
    def accuracy(self, pred, ground_truth):
        print(type(pred))
        print(pred)
        print(type(ground_truth))
        print(ground_truth)
        
        # indices = np.arange(X.shape[0])
        # sections = np.arange(self.batch_size, X.shape[0], self.batch_size)
        # batches_indices = np.array_split(indices, sections)

        # pred = np.zeros_like(y)

        # for batch_indices in batches_indices:
        #     batch_X = X[batch_indices]
        #     pred_batch = self.model.predict(batch_X)
        #     pred[batch_indices] = pred_batch
        # return multiclass_accuracy(pred, y)
        
            
        TP = 0
        for i in range(len(pred)):
            if pred[i] == ground_truth[i]:
                TP += 1
        return TP / len(pred)
    
    def plot_history():
        pass

    # def accuracy(self, pred, label):
    #     answer = F.softmax(pred.detach()).numpy().argmax(1) == label.numpy().argmax(1)
    #     return answer.mean()
    

def binary_classification_metrics(prediction, ground_truth):
    TP, TN, FN, FP = 0, 0, 0, 0
    for i in range(len(prediction)):
        if prediction[i] == True and ground_truth[i] == True:
            TP += 1
        elif prediction[i] == False and ground_truth[i] == False:
            TN += 1
        elif prediction[i] == True and ground_truth[i] == False:
            FN += 1
        elif prediction[i] == False and ground_truth[i] == True:
            FP += 1
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    accuracy = (TP + TN)/len(prediction)
    f1 = 2*((precision * recall)/(precision + recall))
    return precision, recall, f1, accuracy


def make_optimizer(options, model_params):
    params = options['optimizer']
    if params['name'] == 'adam':
        optimizer = torch.optim.Adam(model_params, lr=params['lr'], betas=(params['beta1'], params['beta2']))
    else:
        raise NotImplementedError(f'optimizer {params["optimizer"]} is not implemented')
    return optimizer

