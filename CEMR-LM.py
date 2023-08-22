import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer

class Config(object):
    def __init__(self, dataset):
        self.model_name = 'CEMR-LM'                                      # 模型名称
        self.train_path = dataset + '/data/train.txt'                          # 训练集路径
        self.dev_path = dataset + '/data/dev.txt'                              # 验证集路径
        self.test_path = dataset + '/data/test.txt'                            # 测试集路径
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                          # 类别名单
        self.save_path = dataset + '/save_model/' + self.model_name + '.ckpt'  # 模型训练结果保存路径
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备选择，GPU优先，若无GPU则使用CPU

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 5                                             # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 128                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 3e-5                                       # 学习率
        self.bert_path = './Model'       # BERT模型路径
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)   # BERT分词器
        self.hidden_size = 768                                          # BERT模型输出的隐藏状态维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量（通道数）
        self.dropout = 0.3                                              # Dropout率
        
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)              # 加载预训练的BERT模型
        for param in self.bert.parameters():
            param.requires_grad = True                                      # 设置BERT模型的参数可训练
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_sizes])  # 定义多个卷积层
        self.dropout = nn.Dropout(config.dropout)                           # 定义Dropout层

        self.attention_fc = nn.Linear(config.hidden_size, 1)                # 定义全连接层，将BERT模型的输出进行线性变换
        self.fc_cnn = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)  # 第一个全连接层
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)                                      # 通过卷积层和ReLU激活函数进行卷积操作，并压缩维度
        x = F.max_pool1d(x, x.size(2)).squeeze(2)                           # 使用最大池化进行特征提取，并压缩维度
        return x

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)  # 使用BERT模型进行编码
        # print("encoder_out", encoder_out.shape) #[32,128,768]
        out = encoder_out.unsqueeze(1)                                      # 在第1维度添加一个维度，用于卷积操作
        # print("out1", out.shape) #[32, 1, 128, 768]
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)  # 多个卷积层的操作结果拼接
        out1 = self.dropout(out)
        attention_scores = self.attention_fc(encoder_out).squeeze(-1)       # [batch_size, seq_len]线性变换得到注意力得分
        # print("attention_scores", attention_scores.shape) #[32]
        attention_weights = F.softmax(attention_scores, dim=-1).unsqueeze(-1)  # [batch_size, seq_len, 1]softmax操作得到注意力权重
        # print("attention_weights", attention_weights.shape) #[32, 1]
        attended_out = torch.sum(encoder_out * attention_weights, dim=1)    # [batch_size, hidden_size]注意力加权后的输出
        out = self.dropout(out1+attended_out)                                             # Dropout操作
        out = self.fc_cnn(out)                                      # 第一个全连接层
        return out



