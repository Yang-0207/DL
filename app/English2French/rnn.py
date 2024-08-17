import torch
import torch.nn as nn
import torch.nn.functional as F

# 设备选择，可以选择在GPU上运行或者在CPU上运行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置组成句子中单词或标点的最多个数
DEFAULT_MAX_LENGTH = 10

class EncoderRNN(nn.Module):
    """编码器类"""

    def __init__(self, input_size, hidden_size):
        """
        初始化函数
        :param input_size: 编码器输入尺寸，英文的词表大小
        :param hidden_size: GRU的隐藏层神经单元数，同时也是词嵌入的维度
        """
        super(EncoderRNN, self).__init__()
        # 将参数 hidden_size 传入类中
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 实例化Embedding层，输入参数分别是词表的单词总数，词嵌入的维度
        self.embedding = nn.Embedding(input_size, hidden_size)
        # 实例化GRU
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        """
        :param input: 源语言中的输入张量
        :param hidden: 初始化隐藏层张量
        :return:
        """
        # 经过Embedding处理后，张量是一个二维张量，但是GRU要求输入是三维张量
        # 对结果进行扩展维度 view()，同时让任意单词映射后的尺寸是[1, embedding]
        output = self.embedding(input).view(1, 1, -1)
        # 将output和hidden传入GRU单元中，得到返回结果
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        """初始化隐层张量"""
        # 将隐藏层张量初始化为 1*1*self.hidden_size大小的张量
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        """
        :param hidden_size: 隐藏层神经元个数，同时也是解码器的输入尺寸
        :param output_size: 代表整个解码器的输出尺寸，指定的尺寸也就是目标语言的单词总数
        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        # 实例化Embedding对象，输入参数分别是目标语言的单词总数，和词嵌入的维度
        self.embedding = nn.Embedding(output_size, hidden_size)
        # 实例化GRU对象
        self.gru = nn.GRU(hidden_size, hidden_size)
        # 实例化线性层对象，对GRU的输出做线性变换，得到希望的输出尺寸 output_size
        self.out = nn.Linear(hidden_size, output_size)
        # 最后进入Softmax的处理
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        """
        解码器前向传播
        :param input: 目标语言的输入张量
        :param hidden: 初始化的GRU隐藏层张量
        :return:
        """
        # 经历Embedding层处理后，要将张量形状改变为三维张量
        output = self.embedding(input).view(1, 1, -1)
        # 使用relu函数对输出进行处理，使得Embedding矩阵更稀疏，防止过拟合
        output = F.relu(output)
        # 将张量传入GRU解码器中
        output, hidden = self.gru(output, hidden)
        # 经历GRU处理后的张量是三维张量，但是全连接层需要二维张量，利用output[0]来降维
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        """初始化隐层张量"""
        # 将隐藏层张量初始化为 1*1*self.hidden_size大小的张量
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    """构建基于GRU和Attention的解码器类"""

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=DEFAULT_MAX_LENGTH):
        """
        :param hidden_size: 解码器的GRU输出尺寸，就是隐藏层的神经元个数
        :param output_size: 指定的网络输出尺寸，目标语言的词汇总数（法文）
        :param dropout_p: 使用Dropout层的置零比例
        :param max_length: 句子的最大长度
        """
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # 实例化一个Embedding对象，参数是目标语言的词汇总数喝词嵌入的维度
        self.embedding = nn.Embedding(output_size, hidden_size)

        # 实例化第一个注意力层，注意输入是两个张量的合并
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)

        # 实例化第二个注意力层，注意输入也是两个张量的合并，同时输出要进入GRU中
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # 实例化一个 nn.Dropout层
        self.dropout = nn.Dropout(self.dropout_p)

        # 实例化GRU单元
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)

        # 实例化GRU之后的线性层，作为整个解码器的输出
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input1, hidden, encoder_output):
        """
        :param input1: 元数据的输入张量
        :param hidden: 初始化的隐藏层张量
        :param encoder_output: 代表编码器的输出张量
        :return:
        """
        # 对输入input1进行词嵌入处理，并扩展维度
        embedded = self.embedding(input1).view(1, 1, -1)
        # 紧接着将其输入dropout层，防止过拟合
        embedded = self.dropout(embedded)

        # 在进行第一个注意力层处理前，要将Q,k进行纵轴拼接
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        # 进行bmm操作，注意要将二维张量扩展成三维张量
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_output.unsqueeze(0))

        # 再次进行拼接，顺便要进行一次降维
        output = torch.cat((embedded[0], attn_applied[0]), 1)

        # 将output输入第二个注意力层
        output = self.attn_combine(output).unsqueeze(0)
        # 使用relu进行激活层处理
        output = F.relu(output)

        # 将激活后的张量，连同隐藏层张量，一起传入GRU中

        # 最后将结果先降维，然后线性层梳理成指定的输出维度，最后经过softmax处理
        output = F.log_softmax(self.out(output[0]), dim=1)

        # 返回解码器的最终输出结果，最后的隐藏层张量，注意力权重张量
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)