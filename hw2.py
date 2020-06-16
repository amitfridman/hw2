import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib
import os
import shutil
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
matplotlib.use('agg')
from matplotlib import pyplot as plt


# Hyper Parameters
num_epochs = 20
batch_size = 100
learning_rate = 0.002

# Image Preprocessing
#The following function gets the path of the all pictures together and it splits
#to the two different class labels'
def rearrange_dataset(path):
    files=os.listdir(path)
    if '0' not in files:#we create the two directories once
        os.mkdir(path+'/0')
        os.mkdir(path + '/1')
    else:
        files.remove('0')
        files.remove('1')
    for file in files:
        c=file.split('_')[1].split('.')[0]#we extract the label from the file name
        shutil.move(path+'/'+file, path+'/'+str(c))#we send the pic to the apropiate path.
    return



rearrange_dataset('/home/student/train')
rearrange_dataset('/home/student/test')
transform = transforms.Compose([
    transforms.Resize((48,48)),
    transforms.ToTensor(),


])

transform1 = transforms.Compose([
    #transforms.RandomPerspective(),
    #transforms.RandomAffine(5),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomVerticalFlip(),
    transforms.ColorJitter(),
    transforms.Resize((48,48)),
    transforms.ToTensor(),

])
train_dataset = dsets.ImageFolder('/home/student/train', transform=transform1)
test_dataset = dsets.ImageFolder('/home/student/test', transform=transform)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=2,stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.BatchNorm2d(16))


        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=2,stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.BatchNorm2d(32))

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=2,stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.BatchNorm2d(32))

        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=2,stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.BatchNorm2d(32))

        self.layer6 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=2,stride=2),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.BatchNorm2d(64))


        self.fc = nn.Linear(64, 2)
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer6(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return self.softmax(out)


def to_gpu(x):
    return x.cuda() if torch.cuda.is_available() else x

"""
cnn = CNN()
cnn = to_gpu(cnn)

# convert all the weights tensors to cuda()
# Loss and Optimizer

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate)
print('number of parameters: ', sum(param.numel() for param in cnn.parameters()))
print('Num of trainable parameters :', sum(p.numel() for p in cnn.parameters() if p.requires_grad))
loss_list=[]
train_errors=[]
test_errors=[]
test_loss_list=[]
f1_scores_test=[]
f1_scores_train=[]
train_auc=[]
test_auc=[]
fpr=[]
tpr=[]
fpr_test=[]
tpr_test=[]
correct = 0
total = 0
for epoch in range(num_epochs):
    cnn.train()
    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = to_gpu(images)
            labels = to_gpu(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        outputs = torch.log(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1,
                     len(train_dataset) // batch_size, loss.item()))

    loss_list.append(loss)
    cnn.eval()
    tp = 0.0
    fp = 0.0
    fn = 0.0
    y_score=[]
    y_true=[]
    for i, (images, labels) in enumerate(train_loader):
        images = to_gpu(images)
        labels = to_gpu(labels)
        outputs = cnn(images)
        y_score+=list(outputs.cpu().detach().numpy()[:,1])
        y_true+=list(labels.cpu().detach().numpy())
        outputs = nn.functional.log_softmax(outputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        tp += ((predicted == labels)&(labels==torch.ones(labels.size(),device=torch.device('cuda:0')))).sum()
        fp += ((predicted != labels) & (predicted==torch.ones(labels.size(),device=torch.device('cuda:0')))).sum()
        fn += ((predicted != labels) & (predicted==torch.zeros(labels.size(),device=torch.device('cuda:0')))).sum()
    if epoch==num_epochs-1:
        fpr, tpr, _ = roc_curve(y_true,y_score, pos_label=1)
    train_auc.append(roc_auc_score(y_true,y_score))
    train_errors.append(float((total - correct)) / total)
    f1_scores_train.append(2*tp/(2*tp+fp+fn))
    print(2*tp/(2*tp+fp+fn))
    correct = 0
    total = 0

    tp=0.0
    fp=0.0
    fn=0.0
    y_score = []
    y_true = []
    for images, labels in test_loader:
        images = to_gpu(images)
        labels = to_gpu(labels)
        outputs = cnn(images)
        y_score += list(outputs.cpu().detach().numpy()[:, 1])
        y_true += list(labels.cpu().detach().numpy())
        outputs = nn.functional.log_softmax(outputs)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        #total += labels.size(0)
        correct += (predicted == labels).sum()
        tp += ((predicted == labels) & (labels == torch.ones(labels.size(), device=torch.device('cuda:0')))).sum()
        fp += ((predicted != labels) & (predicted == torch.ones(labels.size(), device=torch.device('cuda:0')))).sum()
        fn += ((predicted != labels) & (predicted == torch.zeros(labels.size(), device=torch.device('cuda:0')))).sum()
    if epoch==num_epochs-1:
        fpr_test, tpr_test, _ = roc_curve(y_true,y_score, pos_label=1)
    test_auc.append(roc_auc_score(y_true,y_score))
    test_errors.append(1-(float(correct) / 6086))
    test_loss_list.append(loss)
    f1_scores_test.append(2*tp/(2*tp+fp+fn))
    correct = 0
    total = 0

plt.plot(range(1, num_epochs + 1), loss_list, label='train')
plt.plot(range(1, num_epochs + 1), test_loss_list, label='test')
plt.title('Loss as Function of Number of Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png')
plt.clf()
plt.plot(range(1, num_epochs + 1), train_errors, label='train')
plt.plot(range(1, num_epochs + 1), test_errors, label='test')
plt.title('Errors as function of Number of Epochs')
plt.xlabel('Epochs')
plt.ylabel('Errors')
plt.legend()
plt.savefig('errors.png')
plt.clf()
plt.plot(range(1, num_epochs + 1), f1_scores_train, label='train')
plt.plot(range(1, num_epochs + 1), f1_scores_test, label='test')
plt.title('F1 as function of Number of Epochs')
plt.xlabel('Epochs')
plt.ylabel('f1')
plt.legend()
plt.savefig('f1.png')
plt.clf()
plt.plot(range(1, num_epochs + 1), train_auc, label='train')
plt.plot(range(1, num_epochs + 1), test_auc, label='test')
plt.title('ROC AUC as function of Number of Epochs')
plt.xlabel('Epochs')
plt.ylabel('ROC AUC')
plt.legend()
plt.savefig('auc.png')
plt.clf()
plt.plot(fpr, tpr, label='train')
plt.plot(fpr_test, tpr_test, label='test')
plt.title('ROC Curve')
plt.xlabel('Epochs')
plt.ylabel('ROC')
plt.legend()
plt.savefig('roc1.png')
# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn1.pkl')
"""

