class ModelConfig:
    batch_size = 64  # 样本大小
    batch_size_pred = 100  # 预测集样本大小
    output_size = 1  # 输出维度
    hidden_dim = 128  # 256/2，神经单元个数
    train_seq_len = 30  # 训练集句子长度
    test_seq_len = 45  # 测试集句长
    embed = 300  # 词向量维度
    n_layers = 2  # lstm层数
    dropout = 0.5  # 过拟合保留比例
    bidirectional = True  # 这里为True，为双向LSTM
    # 训练参数设置
    epochs = 300  # 训练次数
    lr = 0.0003  # 学习率，梯度大小
    print_every = 20  # 每设定的步长，打印输出损失率
    input_path_emd = r'D:\downloan\sgns.weibo.word.bin'  # 预训练词汇
    input_path_pred = r'D:\downloan\my\test1.txt'  # 待预测文件
    input_path_train1 = r'D:\a-数据集-bert\test-wu1.txt'  # 情感训练集文件
    input_path_train2 = r'D:\a-数据集-bert\result_train2.txt'  # 谣言训练集文件
    input_path_test =  r'D:\a-数据集-bert\test-wu1.txt'  # 测试集文件
    input_path_all = r'D:\downloan\ChnSentiCorp\ChnSentiCorp1\train1.txt'  # 训练集和测试集在一个文件
    save_model_path1 = r'C:\Users\ASUS\pythonProject2\bert\sentiment_classifier1[focalloss+adam].tar'
    save_model_path2 = r'C:\Users\ASUS\pythonProject2\bert\sentiment_classifier6.tar'  # 模型保存路径
    save_dict_path = r'D:\downloan\my\zidan'  # 字典保存路径，npy文件
    save_pred_path = r'C:\Users\ASUS\pythonProject2\bert\predict_out0.xlsx'  # 预测结果保存路径，excel


dic={'anger':0, 'happiness':1, 'nerve':2, 'surprise':3, 'sadness':4, 'fear':5, 'like':6, 'disgust':7}
dic1={}
for i,j in enumerate(dic):
    dic1[dic[j]]=j
print(dic1)
# d.items()返回的是： dict_items([('a', 1), ('c', 3), ('b', 2)])
d_order = sorted(dic.items(), key=lambda x: x[1], reverse=False)  # 按字典集合中，每一个元组的第二个元素排列。
# x相当于字典集合中遍历出来的一个元组。
print(d_order)  # 得到:  [('a', 1), ('b', 2), ('c', 3)]

