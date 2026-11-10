# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
To develop a Convolutional Neural Network (CNN) model using PyTorch to classify fashion images into 10 different clothing categories. The model is trained and evaluated using the Fashion-MNIST dataset to achieve accurate image classification.

## Neural Network Model
<img width="1020" height="745" alt="nn diram" src="https://github.com/user-attachments/assets/1e293395-af2d-4077-a110-8a14ead96a0c" />

## DESIGN STEPS
### STEP 1: 

Import PyTorch, Torchvision, NumPy, Matplotlib and Scikit-learn libraries.

### STEP 2: 

Load the Fashion-MNIST dataset and apply tensor conversion and normalization.

### STEP 3: 

Split the dataset into training and testing sets and create DataLoaders with batch size 32.

### STEP 4: 

Build a CNN architecture with three convolutional layers, max pooling and three fully connected layers.

### STEP 5: 

Train the CNN model for 3 epochs using Adam optimizer and CrossEntropyLoss.

### STEP 6: 

Evaluate accuracy, generate a confusion matrix and classification report, and visualize predictions.


## PROGRAM

### Name: JANAGIRAMAN M

### Register Number: 212224230101

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
```
```python
transform=transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.5,),(0.5))  
])
```
```python
train_set=torchvision.datasets.FashionMNIST(root='./data',train=True,download=True,
transform=transform)
test_set=torchvision.datasets.FashionMNIST(root='./data',train=False,download=True,
transform=transform)
```
```python
im,lbl=train_set[0]
print(im.shape)
print(len(train_set))
print(len(test_set))
```
```python
trl=DataLoader(train_set,batch_size=32,shuffle=True)
tstl=DataLoader(test_set,batch_size=32,shuffle=False)
```
```python
class CNNclassifier1(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.c2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.c3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.l1=nn.Linear(128*3*3,64)
        self.l2=nn.Linear(64,32)
        self.l3=nn.Linear(32,10)
    def forward(self,x):
        x=self.pool(torch.relu(self.c1(x)))
        x=self.pool(torch.relu(self.c2(x)))
        x=self.pool(torch.relu(self.c3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.l1(x))
        x=torch.relu(self.l2(x))
        x=self.l3(x)
        return x
```
```python
from torchsummary import summary

model=CNNclassifier1()
criterion=nn.CrossEntropyLoss()
op=optim.Adam(model.parameters(),lr=0.001)
if torch.cuda.is_available():
    print(torch.cuda.is_available())
    device=torch.device('cuda')
    model.to(device)


summary(model,input_size=(1,28,28))

```
```python
epochs=3
running_loss=0.0

for i in range(epochs):
    model.train()
    for a,b in trl:
        op.zero_grad()
        pred=model(a)
        loss=criterion(pred,b)
        loss.backward()
        op.step()
        running_loss+=loss.item()
    print(f"Loss:{i}",running_loss/len(trl))
```
```python
t=0
c=0
act=[]
pre=[]
model.eval()
with torch.no_grad():
    for img,labels in tstl:
        output=model(img)
        _,predicted=torch.max(output,1)
        t=t+labels.size(0)
        c+=(predicted==labels).sum().item()
        pre.extend(predicted.cpu().numpy())
        act.extend(labels.cpu().numpy())
accuracy=c/t*100
print("Accuracy Score:",accuracy)
conf_matrix=confusion_matrix(act,pre)
class_report=classification_report(act,pre,target_names=test_set.classes)
print("Classification Report:",class_report)
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='Blues',xticklabels=test_set.classes,yticklabels=test_set.classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```
```python
with torch.no_grad():
    img1,label=test_set[0]
    output=model(img1.unsqueeze(0))
    _,pred=torch.max(output,1)
    classes=test_set.classes
    img1 = img1 * 0.5 + 0.5
    plt.imshow(img1.squeeze(),cmap="gray")
    plt.title("Predicted Image")
    plt.axis('off')
    plt.show()
    print(f"Actual:{classes[label]}\n")
    print(f"Predicted:{classes[pred.item()]}\n")

```


### OUTPUT

## Training Loss per Epoch

<img width="501" height="67" alt="image" src="https://github.com/user-attachments/assets/5692837c-9804-43cf-bffe-b049a0866bd5" />

## Confusion Matrix

<img width="652" height="553" alt="image" src="https://github.com/user-attachments/assets/e881237b-6adb-49d6-b379-198354fcf5a8" />

## Classification Report
<img width="652" height="362" alt="image" src="https://github.com/user-attachments/assets/c06e96ab-b4dc-4442-a2bf-f55e5107afea" />

### New Sample Data Prediction
<img width="537" height="637" alt="image" src="https://github.com/user-attachments/assets/c7db495f-92b3-4a73-94c9-df5eb41d0132" />

## RESULT
The CNN model successfully classified fashion images into 10 categories using the Fashion-MNIST dataset.
