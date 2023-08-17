import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class ASLDataset(torch.utils.data.Dataset):
    def __init__(self, train_path, test_path=None, transforms=None, imsize=64):
        self.root = train_path
        self.classes = {}
        self.image = []
        self.size = imsize
        self.train = True
        self.transforms = transforms
        listdir = sorted(os.listdir(train_path))
        if test_path:
            for i, name in enumerate(listdir):
                self.classes[name] = i
            self.image = sorted(os.listdir(test_path))
            self.root = test_path
            self.train = False
        else:
            for i, name in enumerate(listdir):
                self.classes[name] = i
                list_class_image = os.listdir(os.path.join(train_path, name))
                self.image += list_class_image
                print(f'{name}:{len(list_class_image)}', end=' | ')

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        name = self.image[idx]
        if name[0] in self.classes:
            class_name = name[0]
            class_num = self.classes[name[0]]
        elif 'del' in name:
            class_name = 'del'
            class_num = self.classes['del']
        elif 'nothing' in name:
            class_name = 'nothing'
            class_num = self.classes['nothing']
        elif 'space' in name:
            class_name = 'space'
            class_num = self.classes['space']
        if self.train:
            path = os.path.join(self.root, class_name, name)
        else:
            path = os.path.join(self.root, name)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img/255.0
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        img = img.transpose([2, 0, 1]) # HWC -> CHW
        t_img = torch.from_numpy(img)
        if self.transforms:
            t_img = self.transforms(t_img)
        t_class_num = torch.tensor(class_num)

        return t_img, t_class_num

def show_image(img):
    plt.imshow(img.numpy().transpose([1, 2, 0]))