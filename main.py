import pandas as pd
df_train = pd.read_csv('./train.csv')
print(f'row of df_train = {len(df_train)}')
df_test = pd.read_csv('./test.csv')
print(f'row of df_test = {len(df_test)}')
df_data = pd.concat([df_train, df_test])
print(f'row of df_data = {len(df_data)}')
df_data.head(10) #注意到有些值是NaN，比如表格第6位乘客的年齡
import seaborn as sns
sns.countplot(x=df_data['Sex'], hue=df_data['Survived']) #視覺化，countplot不用給y軸，一定是次數統計，x給了data裡面的Sex，Seaboen會自己分辨表格內的相異值，按照範例就是分為男性和女性
df_data[["Sex", "Survived"]].groupby(["Sex"]).mean().round(4) #數字驗證，挑出data中的Sex和Survived，用.groupby將Sex分為男性組和女姓組，剩下的mean也不用指定，因為剩下Survived，round(4)是四捨五入到小數點後第4位
#因為 Survived 欄位只有 0（死）和 1（活），在只有 0 和 1 的資料中，平均值就是 1（存活）所佔的比例
#例如：如果有 100 個女性，其中 74 個活下來（1），26 個遇難（0），總和是 74，平均值就是0.74（代表 74% 存活率）。
#資料列表中有NaN，但在畫圖時只要沒給值就不會畫進去，所以不影響作圖
df_data["Sex_Int"] = df_data["Sex"].map({"male": 1, "female": 0}) #df_data["Sex_Int"]如果data中沒有這欄位，pandas會在最右邊新增這欄，所以不一定要表格內有的data
print(df_data[["Sex", "Sex_Int"]])
sns.countplot(x=df_data['Pclass'], hue=df_data['Survived'])
df_data[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().round(4)
#邏輯跟上面Sex差不多，as_index=False就是算完值後，還是保留最左邊的index，as_index=True(預設值)就是把groupby的欄位當作index(跟上面Sex一樣)
# row of df_train = 891
# row of df_test = 418
# row of df_data = 1309
df_train = df_data[:len(df_train)] #資料已經經過處理，要把train和test再分開出來，先切0到891
df_test = df_data[len(df_train):] #剩下的就是原本test的data
df_val = df_train[:100]  #抓train的0到100筆資料
df_train = df_train[100:] #剩下就是要train model的data
print(df_train.shape)
print(df_val.shape)
print(df_test.shape)
#df_val是為了後續調整參數用的(自我比對，假裝是正確答案)，因為df_train全丟給model的話，就沒有正確答案可以對了，也就是說你練完只能丟給kaggle去對答案，連調整參數的機會都沒有，所以才要先把df_train割一點(有正確答案的)
#調整好後丟入df_test，再丟給kaggle
print(df_train["Age"].mean()) #因為表格裡面的年齡有Nan值，計算平均年齡給它，給它30歲
selected_columns = ['Sex_Int', 'Pclass', 'Age'] #挑選這三個參數來做實驗
X_train = df_train[selected_columns].fillna(30).to_numpy() #挑選df_train中我們要的三個參數，並且用fillna()補上平均年齡值，轉換成numpy形式
X_val = df_val[selected_columns].fillna(30).to_numpy()   #作法一樣
Y_train = df_train['Survived'].to_numpy(dtype=int) #Y_train就是正確答案(我們分割出來的)，dtype=int，因為是要看存活，只需要1(存活)或是0(死亡)，不需要小數點
Y_val = df_val['Survived'].to_numpy(dtype=int)   #作法一樣
print(X_train)
print(Y_train)
from torch.utils.data.dataset import Dataset
import torch
class TitanicDataset(Dataset): #包東西給Dataset
  def __init__(self, features, labels, train=False): #三個參數
    super(TitanicDataset, self).__init__()
    self.features = torch.from_numpy(features).float() #給它features轉成torch形式再轉成浮點數
    if train: #有在train的時候就會有labels，沒有在train的話就會跑出-1
      self.labels = torch.from_numpy(labels)
    else:
      self.labels = -1
    self.train = train
  def __getitem__(self, index):
    if self.train:
      return self.features[index], self.labels[index] #train=True回傳遞幾個features和labels資料
    else:
      return self.features[index], -1 #train=False回傳features，labels不回傳
  def __len__(self):#回傳features有多少個，總共有幾個row(幾個乘客)
    return len(self.features)
    from torch.utils.data import DataLoader
batch_size = 100 #data比較少的時候設batch_size沒什麼用，但data量多的時候就要設
# 宣告Dataset
train_dataset = TitanicDataset(X_train, Y_train, train=True)
val_dataset = TitanicDataset(X_val, Y_val, train=True)
test_dataset = TitanicDataset(X_test, None, train=False)
# DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
import torch.nn as nn
class TitanicModel(nn.Module):
  def __init__(self, input_size, output_size): #兩個參數
    super(TitanicModel, self).__init__()
    self.linear = nn.Sequential( #Sequential是容器，將資料接起來，從第一層留到最後一層
        nn.Linear(input_size, 64), #矩陣的概念，features dimension是64
        nn.ReLU(),#大於0的照舊，小於0的變成0
        nn.Linear(64, output_size)
    )

  def forward(self, x):#將x拿進來，跑過剛剛寫好的每一層後，輸出預測答案
    y_pred = self.linear(x)
    return y_pred
import torch.optim as optim
import numpy as np
lr = 0.01
epochs = 50 #跑50次
model = TitanicModel(len(selected_columns), 2)
#model是上面宣告的TitanicModel
#selected_columns = ['Sex_Int', 'Pclass', 'Age']，長度是3
#len(selected_columns)是input_size，output_size是2，因為是分類問題(有無存活)
optimizer = optim.Adam(model.parameters(), lr=lr)
#Adam可以換成SGD或是AdamW都可以試試看
#model.parameters()將參數傳給optimizer去做修正及更新
loss = nn.CrossEntropyLoss()
#因為是分類問題用CrossEntropyLoss()
for epoch in range(epochs):
  # Training
  model.train() # 重要，model要開成train的模式
  train_acc = 0.0
  train_loss = 0.0
  samples = 0
  print(f'Epoch {epoch + 1}:')
  #上面是宣告計算accuracy
  for batch_idx, (x, y) in enumerate(train_loader):
    #batch sampling所以來的資料都會是一個batch_size，batch_size = 100(上面有定義)，從train_loader拿取data
    #batch_idx就是分出來一包一包的batch
    #用x和y接下回傳值(tuple形式)，看getitem那邊，return self.features[index], self.labels[index]，有兩個回傳值
    optimizer.zero_grad() #梯度清空
    y_pred = model(x)  #資料丟到模型算預測值
    batch_loss = loss(y_pred, y) #丟入output和正確答案，預測跟正確答案差多少
    batch_loss.backward() #從loss中，透過backward計算梯度(gradient)
    optimizer.step() #利用optimizer優化參數
    train_acc += np.sum(np.argmax(y_pred.data.numpy(), axis=1) == y.numpy())
    #y_pred是兩個參數的，一定會有一個大一個小，挑出大的index看是多少，之後看這個值跟y(正確答案)是否一樣，回傳True或是Flase，然後把對的相加到train_acc
    train_loss += batch_loss.item() #取出loss的值
    samples += len(x) #實際的data數量
  print(f'train_acc = {train_acc / samples}')  #samples累積處理過的資料筆數，計算平均訓練準確率
  print(f'train_loss = {train_loss / samples}') #計算平均訓練損失

  # Validating，驗證train後的model
  model.eval() # 重要，沒有要train了，要切回eval模式
  val_acc = 0.0
  val_loss = 0.0
  samples = 0
  for batch_idx, (x, y) in enumerate(val_loader):
    y_pred = model(x) #模型對於x預測輸出
    batch_loss = loss(y_pred, y)
    val_acc += np.sum(np.argmax(y_pred.data.numpy(), axis=1) == y.numpy())
    val_loss += batch_loss.item()
    samples += len(x)
  print(f'val_acc = {val_acc / samples}')
  print(f'val_loss = {val_loss / samples}')

  #理想狀態
  #train_loss下降，val_loss也跟著下降
  #train_acc上升，val_acc也跟著上升
  #觀察
  #假設train_loss在20輪持續下降，但是val_loss或是val_acc一直掉，就代表model已經開始死背了，應該要停在第20輪
  model.eval() #重要，要換成Evaluation的模式
prediction = []
for batch_idx, (x, y) in enumerate(test_loader):
  y_pred = model(x)
  prediction.append(list(np.argmax(y_pred.data.numpy(), axis=1)))

answer = prediction[0]
for i in range(1, len(prediction)):
  answer.extend(prediction[i])

#將結果寫入 csv 檔
with open("predict.csv", 'w') as f:
    f.write('PassengerId,Survived\n')
    for i, y in enumerate(answer):
        f.write('{},{}\n'.format(892 + i, y))
