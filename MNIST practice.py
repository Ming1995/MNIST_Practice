import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics.classification import Accuracy
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# 檢查該硬體設備有無GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# downloading training data
train_datasets = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform = transforms.Compose([
        transforms.ColorJitter(hue=(0.3,0.5)),  # 加入色相調整
        ToTensor()
    ])
)
# downloading testing data
test_datasets = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
class_names = train_datasets.classes

# Check out the training sample
# image, label = train_datasets[12]
# print(image,label)

batch_size = 32
train_dataloader = DataLoader(
    train_datasets,
    batch_size=32,
    shuffle=True
)
test_dataloader = DataLoader(
    test_datasets,
    batch_size=32,
    shuffle=True
)
# Let's check out what we've created
# print(f"Dataloaders: {train_dataloader, test_dataloader}") 
# print(f"Length of train dataloader: {len(train_dataloader)} batches of {batch_size}")
# print(f"Length of test dataloader: {len(test_dataloader)} batches of {batch_size}")

#----------------------------
# 原始ReLU()模型
#----------------------------
class FashionMNISTModelV0(nn.Module):
    def __init__(self):
        super(FashionMNISTModelV0,self).__init__()
        self.layer1 = nn.Flatten()
        self.layer2 = nn.Sequential(
            nn.Linear(784,1598),
            nn.BatchNorm1d(1598),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(0.3)
        self.layer3 = nn.Sequential(
            nn.Linear(1598,784),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(32,16,kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer7 = nn.AdaptiveAvgPool2d((1,1))
        self.layer8 = nn.Sequential(
            nn.Linear(16,10)
        )
    def forward(self,x):
        x = self.layer1(x)
        # print(f"After the first layer data shape is {x.shape}")
        x = self.layer2(x)
        x = self.dropout(x)
        # print(f"After the second layer data shape is {x.shape}")
        x = self.layer3(x)
        # print(f"After the thrid layer data shape is {x.shape}")
        x=x.view(-1,1,28,28) #調整成2D格式以便輸入卷積層
        # print(f"After the data reshape is {x.shape}")
        x = self.layer4(x)
        # print(f"After the forth layer data shape is {x.shape}")  # 輸出是([32,32,28,28])
        x = self.layer5(x)
        # print(f"After the fifth layer data shape is {x.shape}")
        x = self.layer6(x)
        # print(f"After the fifth layer data shape is {x.shape}")
        x = self.layer7(x)
        # print(f"After the 6th layer data shape is {x.shape}")
        x = x.squeeze() # 去掉維度為1的維度以便輸入線性層
        x = self.layer8(x)
        # print(f"After the 7th layer data shape is {x.shape}")
        return x
#----------------------------
# 新的LeakyReLU()模型
#----------------------------
class FashionMNISTModelV1(nn.Module):
    def __init__(self):
        super(FashionMNISTModelV1, self).__init__()
        self.layer1 = nn.Flatten()
        self.layer2 = nn.Sequential(
            nn.Linear(784,1598),
            nn.BatchNorm1d(1598),
            nn.LeakyReLU(negative_slope=0.05)  # 修改為 LeakyReLU
        )
        self.dropout = nn.Dropout(0.3)
        self.layer3 = nn.Sequential(
            nn.Linear(1598,784),
            nn.LeakyReLU(negative_slope=0.05)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope=0.05)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(32,16,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope=0.05)
        )
        self.layer6 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer7 = nn.AdaptiveAvgPool2d((1,1))
        self.layer8 = nn.Sequential(
            nn.Linear(16,10)
        )
    def forward(self,x):
        x = self.layer1(x)
        # print(f"After the first layer data shape is {x.shape}")
        x = self.layer2(x)
        x = self.dropout(x)
        # print(f"After the second layer data shape is {x.shape}")
        x = self.layer3(x)
        # print(f"After the thrid layer data shape is {x.shape}")
        x=x.view(-1,1,28,28) #調整成2D格式以便輸入卷積層
        # print(f"After the data reshape is {x.shape}")
        x = self.layer4(x)
        # print(f"After the forth layer data shape is {x.shape}")  # 輸出是([32,32,28,28])
        x = self.layer5(x)
        # print(f"After the fifth layer data shape is {x.shape}")
        x = self.layer6(x)
        # print(f"After the fifth layer data shape is {x.shape}")
        x = self.layer7(x)
        # print(f"After the 6th layer data shape is {x.shape}")
        x = x.squeeze() # 去掉維度為1的維度以便輸入線性層
        x = self.layer8(x)
        # print(f"After the 7th layer data shape is {x.shape}")
        return x
#----------------------------
# 新的PReLU()模型
#----------------------------
class FashionMNISTModelV2(nn.Module):
    def __init__(self):
        super(FashionMNISTModelV2,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer5 = nn.Flatten()
        self.layer6 = nn.Sequential(
            nn.Linear(64*7*7,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,10)     # 一定要增加全連接層將抽取到的特徵映射到最終的類別上
        )
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x
#----------------------------
# 新的V3模型(激活函數activation function 一率改成ReLU())
#----------------------------
class FashionMNISTModelV3(nn.Module):
    def __init__(self):
        super(FashionMNISTModelV3,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer7 = nn.Flatten()
        self.layer8 = nn.Sequential(
            nn.Linear(128*7*7,256),  # 將原本的64改成256增加神經元數量
            nn.ReLU(),
            nn.Dropout(0.5),  # 將原本0.3更正成0.5提高模型泛化能力
            nn.Linear(256,128), # 再添加一個全連接層
            nn.ReLU(),
            nn.Linear(128,10) 
        )
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x
#實例化模型
model_05 = FashionMNISTModelV0()
"""
train_datasets 是一個 torchvision.datasets.FashionMNIST 物件，它不是 Tensor，也不是 DataLoader，而是整個數據集的物件，不能直接送入 model。
"""
images_train,labels_train = next(iter(train_dataloader))
output = model_05(images_train)

images_test,labels_test = next(iter(test_dataloader))
output = model_05(images_test)

# 為訓練階段建立Accuracy的評估器
train_accuracy = Accuracy(task="multiclass",num_classes=10)
# 為測試階段建立Accuracy的評估器
test_accuracy = Accuracy(task="multiclass", num_classes=10)

# # 建立訓練正確率空list
# Training_acc_list = []

# # 建立測試正確率空list
# Testing_acc_list = []

# # 建立迭代次數空list
# Epoch_list = []

# 建立損失函數
loss_function = torch.nn.CrossEntropyLoss()
# 建立優化器
optimizer = torch.optim.Adam(params=model_05.parameters(),lr=0.0005)

y_logits = model_05(images_test)
y_pred_probs = torch.softmax(y_logits,dim=1)
y_preds = torch.argmax(y_pred_probs,dim=1)
# 訓練 ReLU 模型
torch.manual_seed(42)
def train_and_save_model(model, model_name):
    model.to(DEVICE) # 將模型搬到GPU
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0005)
    loss_function = torch.nn.CrossEntropyLoss()
    
    epochs = 20
    for epoch in range(epochs):
        for batch, (X, y) in enumerate(train_dataloader):
            X,y = X.to(DEVICE), y.to(DEVICE) #將數據及標籤搬到GPU
            model.train()
            y_pred = model(X)
            loss = loss_function(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # 儲存模型
    torch.save(model.state_dict(), f"{model_name}.pth")
    print(f"{model_name} 訓練完成，模型已儲存！")

# 訓練並儲存 ReLU 模型
model_05 = FashionMNISTModelV0().to(DEVICE)
train_and_save_model(model_05, "model_05")

# 訓練並儲存 LeakyReLU 模型
model_LeakyReLU = FashionMNISTModelV1().to(DEVICE)
train_and_save_model(model_LeakyReLU, "model_LeakyReLU")

# 訓練並儲存 PReLU()模型
model_PReLU = FashionMNISTModelV2().to(DEVICE)
train_and_save_model(model_PReLU, "model_PReLU")

# 訓練並儲存 V3模型
model_V3 = FashionMNISTModelV3().to(DEVICE)
train_and_save_model(model_V3,"model_V3")



# plt.plot(np.array(Epoch_list),np.array(Training_acc_list),label = "Train acc curve")
# plt.plot(np.array(Epoch_list),np.array(Testing_acc_list),label = "Test acc curve")
# plt.title("Train and Test Acc curve")
# plt.xlabel("Epoch")
# plt.ylabel("Acc")
# plt.xlim(0,epochs)
# plt.ylim(0,1)
# plt.legend()
# plt.show()
"""對模型進行評估，回傳損失與準確度。"""
def eval_model(model, data_loader, loss_fn, accuracy_fn):
    model.to(DEVICE) # 將驗證模型搬到GPU
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        
        for X, y in data_loader:
            X,y = X.to(DEVICE), y.to(DEVICE) #將數據及標籤搬到GPU
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_pred.argmax(dim=1), y)

    loss = loss/ len(data_loader)
    acc =  acc/ len(data_loader)

    return {
        "model_name": model.__class__.__name__,
        "model_loss": loss.item(),
        "model_acc": acc
    }
# 設定評估函數
loss_fn = torch.nn.CrossEntropyLoss()
accuracy_fn = Accuracy(task="multiclass", num_classes=10).to(DEVICE)

# 對 model_05 進行測試評估
model_05_results = eval_model(
    model=model_05,              # 使用 model_05
    data_loader=test_dataloader, # 使用測試集
    loss_fn=loss_fn,             # 使用損失函數
    accuracy_fn=accuracy_fn      # 使用準確度計算函數
)
# 對model_LeakyReLU 進行測試評估
model_LeakyReLU_results = eval_model(
    model=model_LeakyReLU,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn
)
# 對model_PReLU 進行測試評估
model_PReLU_results = eval_model(
    model=model_PReLU,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn
)
# 對model_V3 進行測試評估
model_V3_results = eval_model(
    model=model_V3,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn
)
# 顯示四個模型評估結果
print(model_05_results)
print(model_LeakyReLU_results)
print(model_PReLU_results)
print(model_V3_results)











# # Plot more images
# torch.manual_seed(42)
# fig = plt.figure(figsize=(10,10))
# Rows,Columns = 5,5
# for i in range(Rows*Columns):
#     random_idx = torch.randint(0,len(train_data),size=[1]).item()
#     image,label = train_data[random_idx]
#     fig.add_subplot(Rows,Columns,i+1)
#     plt.imshow(image.squeeze(),cmap="gray")
#     plt.title(class_names[label])
#     plt.axis(False)
#     plt.show()
# # Visualize the 12th picture
# plt.imshow(image.squeeze(0),cmap="gray")
# plt.title(f"Label:{label}")
# plt.show()