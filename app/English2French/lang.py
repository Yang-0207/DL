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
