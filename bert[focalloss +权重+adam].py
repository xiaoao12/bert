import os
from torch.nn import functional as F
import random
import numpy as np
import torch # pytor库，必用
import pandas as pd # pandas库是一个处理各种数据文件的库，类似于wps，可以打开，保存各种word，ppt等格式的文件
import torch.nn as nn #导入nn，这个库里面是一些全连接网络，池化层，这种神经网络中非常基本的一些模块，这里我们主要是用nn.linear是一个全连接层
from matplotlib import pyplot as plt
from transformers import BertModel, BertTokenizer,BertForSequenceClassification# transformer库是一个把各种预训练模型集成在一起的库，导入之后，你就可以选择性的使用自己想用的模型，这里使用的BERT模型。所以导入了bert模型，和bert的分词器，这里是对bert的使用，而不是bert自身的源码，如果有时间或者兴趣的话我会在另一篇文章写bert的源码实现
from sklearn.model_selection import train_test_split #sklearn是一个非常基础的机器学习库，里面都是一切基础工具，类似于聚类算法啊，逻辑回归算法啊，各种对数据处理的方法啊，这里我们使用的train_test_split方法，是把数据，一部分用作训练，一部分用作测试的划分数据的方法
from config import ModelConfig
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
from my import Data_convert
from torch.utils import data

config = ModelConfig()
mask = []
seq_index = []
device = torch.device('cuda:0')
#device = torch.device('cpu')
print('device=', device)
you_path = '../bert-base-chinese'
#同级目录下用 ./  (一个点)
#上一级目录用 ../  (两个点)
print(os.path.isdir(os.path.join(you_path, '')))
class BertMultiTask(nn.Module):
    def __init__(self, num_labels):
        super(BertMultiTask, self).__init__()
        self.model_name = '../bert-base-chinese'
        self.BERT_PATH = '../bert-base-chinese/'
        self.bert = BertModel.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.BERT_PATH)
        self.classifier_task = nn.Linear(768, num_labels).to(device)

    def forward(self, x):
        batch_tokenized = self.tokenizer.batch_encode_plus(
            x,
            add_special_tokens=True,
            max_length=8,
            padding='longest',  # Change this line
            return_tensors='pt',  # Return PyTorch tensors
            truncation=True,
        )  # tokenize、add special token、pad

        input_ids =batch_tokenized['input_ids'].to(device)
        attention_mask =batch_tokenized['attention_mask'].to(device)
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits_task = self.classifier_task(pooled_output)
        return logits_task



class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=3.0, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha.clone().detach().to(device)
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        alpha_t = self.alpha[targets.data.view(-1)]
        F_loss = alpha_t * (1-pt)**self.gamma * CE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss



# 实例化模型

# 定义情感分类任务的输出层
num_labels_sentiment = 8  # 情感分类的类别数
sentiment_classifier = BertMultiTask(num_labels_sentiment)



# 定义训练数据集和数据加载器
data_train1 = Data_convert(config.input_path_train1, config.train_seq_len, config.batch_size)  # 数据训练集实例化
sentence_train1,lable1 = data_train1.count_s2()  # 返回字典，分词句子，标签


sentences1 = sentence_train1
targets1 = lable1

#打乱

# 使用zip函数将特征和标签组合在一起
data6 = list(zip(sentences1, targets1))

# 使用random.shuffle()函数打乱数据集
random.shuffle(data6)

# 再次使用zip函数将特征和标签分开
X_train, y_train = zip(*data6)
# 定义测试数据集和数据加载器
data_train2 = Data_convert(config.input_path_test, config.train_seq_len, config.batch_size)  # 数据训练集实例化
sentence_test,test_lable = data_train2.count_s2()  # 返回字典，分词句子，标签
#


class DataGen(data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

#打包
train_sentiment_dataset=DataGen(X_train,y_train)
test_sentiment_dataset=DataGen(sentence_test,test_lable)
train_sentiment_dataloader=data.DataLoader(train_sentiment_dataset,batch_size=32)
test_sentiment_dataloader=data.DataLoader(test_sentiment_dataset,batch_size=32)

# 定义损失函数和优化器
# 初始化优化器

# 初始化优化器
optimizer = torch.optim.Adam(list(sentiment_classifier.parameters()),lr=2e-5)



# 训练循环

sentiment_classifier.to(device)
num_epochs=2
loss1=[]
class_weights = torch.ones(num_labels_sentiment).to(device)
loss3=[]
# 训练循环
for epoch in range(num_epochs):
    start_time = time.time()

    # 记录开始时间
    criterion = FocalLoss(alpha=class_weights, gamma=0.2, reduce=True)
    loss_sum = 0.0
    accu1 = 0
    # 初始化样本数量
    num_samples1 = 0
    # 先训练情感分类任务10次
    sentiment_classifier.train()
    for step, (token_ids, label) in enumerate(train_sentiment_dataloader):

            inputs1 = token_ids
            label1 = torch.Tensor(label).to(device)
            # 前向传播 - 情感分类任务

            outputs_sentiment = sentiment_classifier(inputs1)

            loss_sentiment = criterion(outputs_sentiment, label1).to(device)

            # 记录分类的损失，用于绘图
            loss1.append(loss_sentiment.cpu().data.numpy())
            loss_sum += loss_sentiment.cpu().data.numpy()

            # 更新情感分类模型的参数
            optimizer.zero_grad()
            loss_sentiment.backward()
            optimizer.step()

            # 每五步输出一次损失
            if (step + 1) % 5 == 0:
                print(f"情感分类：Step {step + 1}, Loss: {format(loss_sentiment.cpu().data.numpy(),'.6f')}")
            # 计算准确度
            accu1 += (outputs_sentiment.argmax(1) == label1).sum().cpu().data.numpy()
            num_samples1 += label1.size(0)

    # 计算平均准确率
    avg_accu1 = accu1 / num_samples1
    avg_loss = loss_sum / (num_samples1)
    end_time = time.time()  # 记录结束时间
    print(f"Epoch {epoch+1}, 平均 Loss: {avg_loss}, 消耗了时间为:  {format((end_time - start_time)/60, '.3f')} 分钟")
    print(f"Epoch {epoch+1}, Average Accuracy for sentiment task: {avg_accu1}")
    # 记录每个epoch的平均损失，用于绘图
    loss3.append(avg_loss)
#测试
    test_loss_sum1 = 0.0
    test_accu1 = 0
    num_samples2 = 0
    total = torch.zeros(num_labels_sentiment)
    correct = torch.zeros(num_labels_sentiment)
    sentiment_classifier.eval()
    for step, (token_ids, label) in enumerate(test_sentiment_dataloader):
        inputs1 = token_ids
        label1 = torch.Tensor(label).to(device)
        num_samples2 += label1.size(0)
        with torch.no_grad():
            out = sentiment_classifier(inputs1)
            loss = criterion(out, label1)
            test_loss_sum1 += loss.cpu().data.numpy()
            test_accu1 += (out.argmax(1) == label1).sum().cpu().data.numpy()
            predictions = torch.argmax(out, dim=-1)
            for k in range(num_labels_sentiment):
                total[k] += (label1 == k).sum().item()
                correct[k] += ((label1 == k) & (predictions == k)).sum().item()
    accuracies = correct / total
    # 根据每个类别的分类准确率来更新样本权重
    class_weights = (accuracies + 1e-3)
    print("epoch % d,test_sentiment_loss:%f,test_sentiment_acc:%f" % (epoch,test_loss_sum1 / len(test_sentiment_dataloader),
        test_accu1 / num_samples2))
# 使用BERT模型进行预测
sentiment_classifier.eval()
predictions = []
labels=[]
with torch.no_grad():
    for step, (token_ids, label) in enumerate(test_sentiment_dataloader):
        inputs1 = token_ids
        out = sentiment_classifier(inputs1)
        predictions.extend(torch.argmax(out, dim=-1).tolist())
        labels.extend(label.tolist())

from collections import Counter

# 使用Counter来统计每个类别的预测数量
prediction_counts = Counter(predictions)

# 打印每个类别的预测数量
print(prediction_counts)

# 检查是否有类别的预测数量为零
for label in range(num_labels_sentiment):  # 假设num_classes是类别的总数
    if prediction_counts[label] == 0:
        print(f'类别 {label} 没有被预测到任何样本。')

# 计算评估指标
accuracy = accuracy_score(labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro',zero_division=1)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

#模型保存
torch.save(sentiment_classifier.state_dict(), config.save_model_path1)

# 绘制损失图

# 创建一个新的图形
plt.figure()

# 绘制情感分类损失
plt.plot(loss1, label='Sentiment Loss')

# 绘制总损失
plt.plot(loss3, label='Total Loss')

# 添加图例
plt.legend()

# 添加标题和标签
plt.title('Training Loss over Time')
plt.xlabel('Iteration')
plt.ylabel('Loss')

# 显示图形
plt.show()






