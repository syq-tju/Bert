<<<<<<< HEAD
#pip install transformers torch scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split

# 加载数据
data = pd.read_csv('./imdbs.csv')  # 使用正确的路径
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.1, random_state=42)

from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch

# 初始化 tokenizer 和 model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenization
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=512)

# 转换为 torch dataset
train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']), torch.tensor(train_encodings['attention_mask']), torch.tensor(y_train.values))
test_dataset = TensorDataset(torch.tensor(test_encodings['input_ids']), torch.tensor(test_encodings['attention_mask']), torch.tensor(y_test.values))

from torch.optim import AdamW
from transformers import get_scheduler
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
optim = AdamW(model.parameters(), lr=5e-5)

num_epochs = 10
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optim, num_warmup_steps=0, num_training_steps=num_training_steps)

model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        # 更新这里的批次处理方式
        input_ids, attention_mask, labels = [tensor.to(device) for tensor in batch]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        optim.step()
        lr_scheduler.step()
        optim.zero_grad()

        print(f"Epoch: {epoch}, Loss: {loss.item()}")

from sklearn.metrics import accuracy_score

model.eval()
predictions, true_labels = [], []
with torch.no_grad():
    for batch in DataLoader(test_dataset, batch_size=8):
        # 同样地，对批次数据进行解包，并发送到相应设备
        input_ids, attention_mask, labels = [tensor.to(device) for tensor in batch]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=-1).tolist())
        true_labels.extend(labels.tolist())

accuracy = accuracy_score(true_labels, predictions)
=======
#pip install transformers torch scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split

# 加载数据
data = pd.read_csv('./imdbs.csv')  # 使用正确的路径
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.1, random_state=42)

from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch

# 初始化 tokenizer 和 model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenization
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=512)

# 转换为 torch dataset
train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']), torch.tensor(train_encodings['attention_mask']), torch.tensor(y_train.values))
test_dataset = TensorDataset(torch.tensor(test_encodings['input_ids']), torch.tensor(test_encodings['attention_mask']), torch.tensor(y_test.values))

from torch.optim import AdamW
from transformers import get_scheduler
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
optim = AdamW(model.parameters(), lr=5e-5)

num_epochs = 10
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optim, num_warmup_steps=0, num_training_steps=num_training_steps)

model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        # 更新这里的批次处理方式
        input_ids, attention_mask, labels = [tensor.to(device) for tensor in batch]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        optim.step()
        lr_scheduler.step()
        optim.zero_grad()

        print(f"Epoch: {epoch}, Loss: {loss.item()}")

from sklearn.metrics import accuracy_score

model.eval()
predictions, true_labels = [], []
with torch.no_grad():
    for batch in DataLoader(test_dataset, batch_size=8):
        # 同样地，对批次数据进行解包，并发送到相应设备
        input_ids, attention_mask, labels = [tensor.to(device) for tensor in batch]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=-1).tolist())
        true_labels.extend(labels.tolist())

accuracy = accuracy_score(true_labels, predictions)
>>>>>>> 8b141dddd94f5b1c71343f8b10c32c446951cc15
print(f"Test Accuracy: {accuracy}")