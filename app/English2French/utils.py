import time
import math


def timeSince(start):
    """
    获得每次打印的训练耗时
    :param since: 训练开始的时间
    :return:
    """
    now = time.time()
    # 计算得到一个时间差
    s = now - start
    # 将s转化为分钟，秒的形式
    m = math.floor(s / 60)
    s -= m * 60
    # 按照指定的格式返回时间差
    return '%dm %ds' % (m, s)


# since = time.time() - 620
#
# period = timeSince(since)
# print(period)