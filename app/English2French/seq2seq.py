import random
import torch
import torch.nn as nn

import time

import matplotlib.pyplot as plt

# 导入优化方法的工具包
from torch import optim

from app.English2French.lang import Lang, readLangs
from app.English2French.rnn import EncoderRNN, DecoderRNN, AttnDecoderRNN
from app.English2French.utils import timeSince

# 设备选择，可以选择在GPU上运行或者在CPU上运行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义起始标志
SOS_token = 0
# 定义结束标志
EOS_token = 1

lang1 = 'eng'
lang2 = 'fra'

# 明确一下数据文件的存放地址
data_path = '../../data/tutorial/eng-fra.txt'

# 设置组成句子中单词或标点的最多个数
MAX_LENGTH = 10


# 设定 teacher_forcing 的比率，在大多数的概率下使用这个策略进行训练
teacher_forcing_ratio = 0.5

# 选择带有指定前缀的英文源语言的语句数据作为训练数据
eng_prefixes = (
    "i am ", "i m",
    "he is", "he s",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(pair):
    """
    过滤语言对
    :param pair: 语言对
        pair[0]表示英文源语句，长度应小于MAX_LENGTH，并且以指定前缀开始
        pair[1]表示法文源语句，长度应小于MAX_LENGTH
    :return:
    """
    return len(pair[0].split(' ')) < MAX_LENGTH and pair[0].startswith(eng_prefixes) and len(
        pair[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    """过滤语言对的函数"""
    # 函数直接遍历列表中的每个语言字符串并调用filterPair()函数即可
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2):
    """
    整合数据预处理的函数
    :param lang1: 源语言的名字，英文
    :param lang2: 目标语言的名字，法文
    :return:
    """
    # 第一步：通过调用 readLangs() 函数得到两个类对象，并得到字符串类型的语言对的列表
    input_lang, output_lang, pairs = readLangs(lang1, lang2)
    # 第二步：对字符串类型的列表进行过滤操作
    pairs = filterPairs(pairs)
    # 第三步： 对过滤后的语言对列表进行遍历操作，添加进类对象中
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData(lang1, lang2)


def tensorFromSentence(lang, sentence):
    """
    将文本句子转化为张量
    :param lang: 传入的Lang的实例化对象
    :param sentence: 传入的语句
    :return:
    """
    # 对句子进行分割并遍历每一个词汇，然后使用lang的word2index方法找到它对应的索引
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    # 列表的最后添加一个句子结束的符号
    indexes.append(EOS_token)
    # 将其封装成 torch.tensor类型，并改变形状为 n * 1
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    """
    :param pair: 语言对（英文、法语）
    :return:
    """
    # 依次调用具体的处理函数，分别处理源语言和目标语言
    input_tensor = tensorFromSentence(input_lang, pair[0])
    output_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, output_tensor)


def train(input_tensor, target_tensor, encoder, decoder,
          encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    """
    训练函数
    :param input_tensor: 代表源语言的输入张量
    :param target_tensor: 代表目标语言的输入张量
    :param encoder: 代表编码器的实例化对象
    :param decoder: 代表解码器的实例化对象
    :param encoder_optimizer: 编码器优化器
    :param decoder_optimizer: 解码器优化器
    :param criterion: 损失函数
    :param max_length: 句子的最大长度
    :return:
    """
    # 初始化编码器的隐藏层张量
    encoder_hidden = encoder.initHidden()
    # 训练前将编码器和解码器的优化器梯度归零
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # 根据源文本和目标文本张量获得对应的长度
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    # 初始化编码器的输出张量，形状是 max_length * encoder.hidden_size
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    # 设置初始损失值
    loss = 0
    # 遍历输入张量
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        # 每一个轮次的输出encoder_output是三维张量，使用 [0,0] 进行降维到以为列表，赋值给输出张量
        encoder_outputs[ei] = encoder_output[0, 0]
    # 初始化解码器的第一个输入字符
    decoder_input = torch.tensor([[SOS_token]], device=device)

    # 初始化解码器的隐藏层张量，赋值给最后一次编码器的隐藏层张量
    decoder_hidden = encoder_hidden

    # 判断是否使用teacher_forcing
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # 如果使用teacher_forcing
    if use_teacher_forcing:
        # 遍历目标张量，进行解码
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # 使用损失函数技术按损失值，并进行累加
            loss += criterion(decoder_output, target_tensor[di])
            # 因为使用了teacher_forcing，所以将下一步的解码器输入强制设定为”正确的答案“
            decoder_input = target_tensor[di]
    # 如果不使用teacher_forcing
    else:
        # 遍历目标张量，进行解码
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # 预测值变成输出张量中概率最大的那个
            topv, topi = decoder_output.topk(1)
            # 使用损失函数计算损失值，并进行累加
            loss += criterion(decoder_output, target_tensor[di])
            # 如果某一步的解码结果是句子终止符号，则解码直接结束，跳出循环
            if topi.squeeze().item() == EOS_token:
                break;
            # 下一步解码器的输入要设定为当前步最大概率值的那一个
            decoder_input = topi.squeeze().detach()
    # 应用反向传播进行梯度计算
    loss.backward()
    # 利用编码器和解码器的优化器进行参数的更新
    encoder_optimizer.step()
    decoder_optimizer.step()

    # 返回平均损失
    return loss.item() / target_length


def trainIter(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    """
    :param encoder: 编码器的实例化对象
    :param decoder: 解码器的实例化对象
    :param n_iters: 训练的总迭代步数
    :param print_every: 每隔多少轮次进行一次训练日志的打印
    :param plot_every: 每隔多少轮次进行一次损失值的添加，为了后续绘制损失曲线
    :param learning_rate: 学习率
    :return:
    """
    # 训练开始时间
    start = time.time()
    # 初始化存放平均损失值的列表
    plot_losses = []
    # 每隔打印时间隔的总损失值
    print_loss_total = 0
    # 每个绘制曲线损失值的列表
    plot_loss_total = 0

    # 定义编码器和解码器的优化器
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # 定义损失函数
    criterion = nn.NLLLoss()

    # 按照设定的总迭代次数进行迭代训练

    print(' %s     %s    %s  %s' % ("耗时", "轮次", "进度", "平均损失"))

    for iter in range(1, n_iters + 1):
        # 从每次语言对的列表中随机抽取一条样本作为本轮迭代的训练数据
        training_pair = tensorsFromPair(random.choice(pairs))
        # 一次将选取出来的语句对作为输入张量和输出张量
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        # 调用 train 函数获得本轮迭代的损失函数
        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        # 将本轮迭代的损失值进行累加
        print_loss_total += loss
        plot_loss_total += loss

        # 如果到达了打印轮次
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            # 为了下一个打印间隔的累加，这里将累加器清零
            print_loss_total = 0
            # 打印若干信息
            print('%s   %d   %d%%    %.4f' % (timeSince(start), iter, iter / n_iters * 100, print_loss_avg))

        # 如果到达了绘制损失曲线的轮次
        if iter % plot_every == 0:
            # 首先获取本次损失添加的平均损失值
            plot_loss_avg = plot_loss_total / plot_every
            # 将平均损失值添加进最后的列表中
            plot_losses.append(plot_loss_avg)
            # 为了下一个提娜佳损失值的累加，这里将累加器清零
            plot_loss_total = 0

    # 绘制损失曲线
    plt.figure()
    plt.plot(plot_losses)
    plt.savefig("./s2s_loss.png")


# 设置隐层大小256 词嵌入维度
hidden_size = 256


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    """
    评估函数
    :param encoder: 编码器对象
    :param decoder: 解码器对象
    :param sentence: 待评估的源语句
    :param max_length: 句子的最大长度
    :return:
    """
    # 梯度不进行改变
    with torch.no_grad():
        # 对输入的句子进行张量表示
        input_tensor = tensorFromSentence(input_lang, sentence)
        # 获得输入的句子长度
        input_length = input_tensor.size(0)
        # 初始化编码器的隐藏层张量
        encoder_hidden = encoder.initHidden()

        # 初始化编码器的输出张量，矩阵的形状 max_length * hidden_size
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        # 遍历输入张量
        for ei in range(input_length):
            # 循环进入编码器的处理
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            # 将三维张量的输出进行降维到一维，然后赋值给encoder_outputs
            encoder_outputs[ei] = encoder_output[0, 0]

        # 初始化解码器的第一个输入，起始字符
        decoder_input = torch.tensor([[SOS_token]], device=device)
        # 初始化解码器的隐藏层输入
        decoder_hidden = encoder_hidden

        # 初始化预测词汇的列表
        decoded_words = []
        # 初始化一个 attention 张量
        decoder_attentions = torch.zeros(max_length, max_length)

        # 遍历解码
        for di in range(max_length):
            # 将张量送入解码器处理
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # 首先将注意力张量保存
            decoder_attentions[di] = decoder_attention.data
            # 按照解码器输出的最高概率作为当前时间步的预测值
            topv, topi = decoder_output.topk(1)
            # 如果解析出的是结束字符
            if topi.item() == EOS_token:
                # 将结束字符添加到结果列表中，并退出解码循环
                decoded_words.append('<EOS>')
                break
            else:
                # 要根据索引去将真是的字符添加进结果列表中
                decoded_words.append(output_lang.index2word[topi.item()])

            # 最后一步预测的标签赋值给下一步解码器的输入
            decoder_input = topi.squeeze().detach()

        # 返回最终解码的结果列表，以及注意力张量
        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=6):
    """
    :param encoder:
    :param decoder:
    :param n:
    :return:
    """
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print(' ')


def translate(sentence):
    output_words, attention = evaluate(encoder, attn_decoder, sentence)
    output_sentence = ' '.join(output_words)
    return output_sentence, attention


if __name__ == '__main__':
    # 迭代次数
    iters = 5000
    # 输出间隔
    print_every = 500

    # 编码器
    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    # 带有注意力机制的解码器
    attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    # 训练
    trainIter(encoder, attn_decoder, iters, print_every=print_every)
    # 随机评估
    evaluateRandomly(encoder, attn_decoder)

    # 使用
    sentence = "we re both teachers ."
    output_sentence, attention = translate(sentence)
    print(output_sentence)

    plt.figure()
    plt.matshow(attention.numpy())
    plt.savefig("./s2s_attn.png")
