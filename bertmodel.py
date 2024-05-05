from transformers import BertModel, BertTokenizer,BertForSequenceClassification# transformer库是一个把各种预训练模型集成在一起的库，导入之后，你就可以选择性的使用自己想用的模型，这里使用的BERT模型。所以导入了bert模型，和bert的分词器，这里是对bert的使用，而不是bert自身的源码，如果有时间或者兴趣的话我会在另一篇文章写bert的源码实现
from sklearn.model_selection import train_test_split #sklearn是一个非常基础的机器学习库，里面都是一切基础工具，类似于聚类算法啊，逻辑回归算法啊，各种对数据处理的方法啊，这里我们使用的train_test_split方法，是把数据，一部分用作训练，一部分用作测试的划分数据的方法
import torch.nn as nn #导入nn，这个库里面是一些全连接网络，池化层，这种神经网络中非常基本的一些模块，这里我们主要是用nn.linear是一个全连接层
import torch # pytor库，必用


device = torch.device('cpu')

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
            max_length=100,
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
