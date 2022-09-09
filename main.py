import os.path
import shutil

import torch
import torch.nn as nn
import torchvision.datasets
from torchvision import transforms
import os

class Unit(nn.Module):
    def __init__(self,inc,ouc):
        super(Unit, self).__init__()
        self.unit_net=nn.Sequential(nn.Conv2d(inc,ouc,kernel_size=3,padding=0),
                                    nn.BatchNorm2d(ouc),
                                    nn.ReLU())

    def forward(self,x):
        return self.unit_net(x)



class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.net=nn.Sequential(Unit(1,32),
                               Unit (32,32),
                               Unit(32,32),

                               nn.MaxPool2d(2),

                               Unit(32,64),
                               Unit(64,64),
                               Unit(64,64),
                               Unit(64,64),

                               nn.MaxPool2d(2),

                               Unit(64,128),
                               Unit(128,128),
                               Unit(128,128),
                               Unit(128,128),

                               nn.MaxPool2d(2),

                               Unit(128,128),
                               Unit(128,128),
                               Unit(128,128),

                               nn.AvgPool2d(4)
                               )

        self.fc=nn.Linear(128,10)

    def forward(self,x):
        y=self.net(x)
        y=y.view(-1,128)
        return self.fc(y)


train_transforms=transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
    ]
)

train_set=torchvision.datasets.MNIST(root='data',train=True,transform=train_transforms,download=True)
train_dataloader=torch.utils.data.DataLoader(train_set,batch_size=512,shuffle=True)


#测试集的转换
test_transforms=transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])

#下载测试集
test_set=torchvision.datasets.MNIST('data',train=False,transform=test_transforms,download=True)

test_dataloader=torch.utils.data.DataLoader(test_set,batch_size=512,shuffle=False)

CUDA=torch.cuda.is_available()

module=Net()

if CUDA:
    module.cuda()

optimizer=torch.optim.Adam(module.parameters())

loss_f=nn.CrossEntropyLoss()


param_path=r'./param/mnist_cnn.pkl'
tmp_param_path=r'./param/mnist_cnn_temp.pkl'

def adjust_lr(epoch):
    lr=0.001

    if epoch>180:
        lr = lr / 1000000
    elif epoch>150:
        lr = lr / 100000
    elif epoch>120:
        lr = lr / 10000
    elif epoch>90:
        lr = lr / 1000
    elif epoch>60:
        lr = lr / 100
    elif epoch>30:
        lr = lr / 10

    for para_group in optimizer.param_groups:
        para_group['lr']=lr

def test():
    test_acc=0
    module.eval()

    for j,(imgs,labels) in enumerate(test_dataloader):
        if CUDA:
            imgs=imgs.cuda()
            labels=labels.cuda()

        outs=module(imgs)
        _,prediction=torch.max(outs,1)
        test_acc+=torch.sum(prediction==labels)
    test_acc=test_acc.cpu.item()/10000
    return test_acc




def train(num_epoch):

    if os.path.exists(param_path):
        module.load_state_dict(torch.load(param_path))

    for epoch in range(num_epoch):
        train_loss=0
        train_acc=0
        module.train()

        for i,(imgs,labels) in enumerate(train_dataloader):
            if CUDA:
                imgs=imgs.cuda()
                labels=labels.cuda()


            outs=module(imgs)

            loss=loss_f(outs,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss+=loss.cpu().item()*imgs.size(0)

            _,prediction=torch.max(outs,1)
            train_acc+=torch.sum(prediction==labels)

    adjust_lr(epoch)
    train_loss=train_loss/60000
    train_acc=train_acc.cpu().item()/60000

    test_acc=test()
    best_acc=0

    if test_acc>best_acc:
        best_acc=test_acc
        if os.path.exists(tmp_param_path):
            shutil.copyfile(tmp_param_path,param_path)

        torch.save(module.state_dict(),tmp_param_path)

    print('epoch==',epoch,' train_acc==',train_acc,'test_acc==',test_acc)






train(1000)












