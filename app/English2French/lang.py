import re
# 用于字符规范化
import unicodedata
from io import open

data_path = '../../data/tutorial/eng-fra.txt'

class Lang():
    """语言类"""
    def __init__(self, name):
        """
        :param name: 传入某种语言的名字
        """
        self.name = name
        # 初始化单词到索引的映射字典
        self.word2index = {}
        # 初始化索引到单词的映射字典，其中0，1对应SOS，EOS已经在字典中了
        self.index2word = {0: "SOS", 1: "EOS"}
        # 初始化词汇对应的数字索引，从2开始，因为0，1已经被开始字符和结束字符占用了
        self.n_words = 2

    def addWord(self, word):
        """
        添加单词到类内字典中，将单词转化为数字
        """
        if word not in self.word2index:
            # 添加的时候，索引值取当前类中单词的总数
            self.word2index[word] = self.n_words
            # 再添加反转的字典
            self.index2word[self.n_words] = word
            # 更新类内单词总数量
            self.n_words += 1

    def addSentence(self, sentence):
        """
        添加句子函数，将整个句子中所有的单词一次添加到字典中
        因为英文、法文都是空格进行分割的语言，直接进行分词就可以
        :param sentence:
        :return:
        """
        for word in sentence.split(' '):
            self.addWord(word)


def readLangs(lang1, lang2):
    """
    读取原始数据并实例化源语言 + 目标语言的类对象
    :param lang1:
    :param lang2:
    :return:
    """
    lines = open(data_path, encoding='utf-8').read().strip().split('\n')
    # 对lines列表中的句子进行标准化处理，并以 \t 进行再次划分，形成子列表
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    # 直接初始化两个类对象
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    return input_lang, output_lang, pairs

def normalizeString(s):
    """字符串规范化函数"""
    # 使字符串转变为小写并去除掉两侧的空白符，再调用上面的函数转换为ASCII字符串
    s = unicodeToAscii(s.lower().strip())
    # 在.！？前面加一个空格
    s = re.sub(r"([.!?])", r" \1", s)
    # 使用正则表达式将字符串中不是大小写字符和正常标点符号的全部替换为空格
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# 将 unicode 字符串转换为 ASCII 字符串，主要用于将法文的重音符号去除掉
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')