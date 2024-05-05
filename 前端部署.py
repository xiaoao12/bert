import streamlit as st
from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from bertmodel import  BertMultiTask
from config import ModelConfig#配置

config=ModelConfig()#实例化配置
# 加载模型
model = BertMultiTask(8)  # 假设我们有两个分类标签
model.load_state_dict(torch.load(config.save_model_path1, map_location=torch.device('cpu')))
model.eval()


# Streamlit应用
st.title('文本分类器')

# 用户输入文本
user_input = st.text_area("请输入文本", "这里是示例文本")

# 对输入文本进行预测
def predict(text):
    text=[text]
    with torch.no_grad():
        predictions = model(text)
        predictions=predictions.argmax(dim=1).item()
        dic = {4: 'sadness', 1: 'happiness', 2: 'nerve', 6: 'like', 0: 'anger', 5: 'fear', 3: 'surprise', 7: 'disgust'}
    return dic[predictions]

if st.button('预测'):
    label = predict(user_input)
    st.write(f'预测结果: {label}')


