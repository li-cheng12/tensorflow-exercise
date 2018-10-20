# -*- coding: utf8 -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic word2vec example."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib

matplotlib.use('Agg')  # jason
import collections
import math
import os
import random
from tempfile import gettempdir
import zipfile
import pdb
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'


# pylint: disable=redefined-outer-name
# 下载并验证text8数据集
def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    local_filename = os.path.join(gettempdir(), filename)
    # 先看是否已经下载过了
    if not os.path.exists(local_filename):
        local_filename, _ = urllib.request.urlretrieve(url + filename,
                                                       local_filename)
    statinfo = os.stat(local_filename)
    # 校验文件的尺寸
    print('size:', statinfo.st_size)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + local_filename +
                        '. Can you get to it with a browser?')
    return local_filename


# pdb.set_trace()
# filename = maybe_download('text8.zip', 31344016)


# Read the data into a list of strings.
# 读取数据集，转化为列表vocabulary(每个元素为单词)
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    # 解压文件
    with zipfile.ZipFile(filename) as f:
        # tf.compat.as_str将数据转换成单词的列表
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


# 读取词库, vocabulary是词的列表，实际也是一个一个的句子，就是训练数据集，将所有的这些词也用来构成词典
# vocabulary = read_data(filename)
vocabulary = tf.compat.as_str(open('/home/licheng/text8', 'r').read()).split()
print('Data size', len(vocabulary))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))  # most_common：返回计数值最高的top n个,结果已经按词频降序排列
    dictionary = dict()
    # 将所有的词典的词排个顺序，假设词典有这些词： 我，你，他，和，还。编个顺序:我(1),你(2),他(3),和(4),还(5)
    # 那么将'我'转换成one-hot向量就是[1,0,0,0,0,], '他'转换成one-hot向量就是[0,0,1,0,0]
    for word, _ in count:
        dictionary[word] = len(dictionary)  # len(dictionary)就是当前的词word的顺序, 词频最高的单词的编码为0，其次为1，依次增加
    data = list()
    unk_count = 0
    for word in words:
        # count中只有top的那些词，所以dictionary中也是只有top的那些词，所以对于非top的词index=0，
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1  # unk_count是非top的词的个数
        data.append(index)
    count[0][1] = unk_count  # 所有非top词出现的次数
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    # dictionary: 每个词的顺序值, 词：编码
    # reversed_dictionary：将dictionary中的key和value颠倒过来， 编码：词
    # count： 词典中的词出现的次数，非top的合并起来了
    # data： 词典中词的顺序值, 就是单词的编码, 将读取的训练数据(句子)中的词全部转换成对应的词的编码
    print("data:", data)
    print("count:", count)
    print("dictionary:", dictionary)
    print("reversed_dictionary:", reversed_dictionary)
    return data, count, dictionary, reversed_dictionary


# Filling 4 global variables:
# data - list of codes (integers from 0 to vocabulary_size-1).
#   This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# dictionary - map of words(strings) to their codes(integers)
# reverse_dictionary - maps codes(integers) to words(strings)
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
del vocabulary  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0  # 单词的序号


# Step 3: Function to generate a training batch for the skip-gram model.
# batch_size: 每个批次生成的训练实例个数
# skip_window: 中心词左右的窗口大小,即中心词左边词的个数或者右边词的个数
# num_skips: 对于一个中心词，从其窗口内随机选择的实例个数， 实例的x是中心词语，y是随机采样的上下文词语
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    # batch_size是每个批次要生成的训练实例的个数，而num_skips是每个词的上下文中随机选择的训练实例数，num_skips * k = batch_size， k是选择目标词个数
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ] #整个区域的长度，包括中心词和上下文的词
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):  # data中是单词的编码, 词频最高的单词的编码为0，其次为1，依次增加
        data_index = 0
    buffer.extend(data[data_index:data_index + span])  # buffer用于存储长度为 span 的单词编号
    data_index += span

    for i in range(batch_size // num_skips):  # batch_size // num_skips： 选取目标中心词语的次数，窗口滑动的次数
        context_words = [w for w in range(span) if w != skip_window]  # context_words: 这是一个中心词的上下文词的索引集合
        words_to_use = random.sample(context_words, num_skips)  # 随机从上下文词中选择num_skips个词的索引值返回
        # 生成一个目标单词的训练样本
        for j, context_word in enumerate(words_to_use):
            # 可以看到batch里面的值都是一样的，都是中心词, labels里面是中心词对应的上下文
            # 对应关系是： 中心词-> 上下文-2, 中心词->上下文-1, 中心词->上下文+1， 中心词->上下文+2
            batch[i * num_skips + j] = buffer[skip_window]  # skip_window对应的是中心词的位置, buffer[skip_window]是中心词的编码
            labels[i * num_skips + j, 0] = buffer[context_word]  # buffer[context_word]是上下文词的编码
        if data_index == len(data):
            buffer[:] = data[:span]
            data_index = span
        else:
            buffer.append(data[data_index])  # 由于buffer长度有限，这会将队首的挤出去
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],
          '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
num_sampled = 64  # Number of negative examples to sample. # 训练时用来做负样本的噪声单词的数量

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
# 生成验证数据，随机抽取一些频数最高的单词，看向量空间上跟它们距离最近的单词是否相关性比较高
# 验证单词其实就是我们要从全部单词中找出与验证单词相近的8个单词，所以会计算全部单词的向量与验证单词的向量的相似度
valid_size = 16  # Random set of words to evaluate similarity on. # 抽取的验证单词数
valid_window = 100  # Only pick dev samples in the head of the distribution. # 验证单词只从频数最高的 100 个单词中抽取,
# 从频数最高的valid_window个单词中随机选择valid_size个词作为验证单词
valid_examples = np.random.choice(valid_window, valid_size, replace=False)  # 从(0,valid_window)的区间随机选择valid_size个数字返回

graph = tf.Graph()

with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])  # 1维, [1,2]
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])  # 2维 [[1], [2]]
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)  # valid_dataset： 是验证数据集

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        # embedding其实就是模型中间的那个维度, 就是输入和输出之间的那个维度，就是所谓的投影层
        # 这行代码是在构造一个词典
        # embedding_size：是将单词转换为稠密向量的维度
        # 随机生成所有单词的词向量 embeddings，单词表大小 5000，向量维度 128
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0,
                              1.0))  # [vocabulary_size, embedding_size]是形状，-1.0, 1.0是最小值和最大值
        # 从embeddings词典中查找训练输入train_inputs对应的向量,这样就将每个输入表示成了一个向量
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        # 产生一个正态分布,均值mean默认为0, 方差stddev, nce_weights莫非就是输入与投影层之间连线的权重值
        # wx + b, 这里nce_weights应该就是w，nce_biases应该就是b
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    # Explanation of the meaning of NCE loss:
    #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
    # num_sampled是负采样的个数，标签向量只会有一个元素为1，即标签向量肯定是one-hot向量，为1的那一位是正实例，其他为0的是负实例
    # 我们学习的目的主要是使正实例那一位为1，其他的负实例可以随机选择一部分来计算损失函数
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))  # num_classes是分类的个数，所以可以认为是词典中词的个数

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))  # 可以认为是词典的长度
    normalized_embeddings = embeddings / norm  # 标准化词典, embeddings是所有单词的词向量,normalized_embeddings是所有单词的词向量的标准化形式,即长度为1
    valid_embeddings = tf.nn.embedding_lookup(  # valid_embeddings是验证数据集单词对应的嵌套向量(嵌套向量其实就是一个向量)
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)  # 这可以认为是在计算验证词向量和所有词向量的相似度

    # Add variable initializer. #初始化tensorflow向量，并激活
    init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 100001  # 最大的迭代训练次数

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    average_loss = 0
    for step in xrange(num_steps):
        # 准备每个批次的训练数据
        # 每次训练后，调整normalized_embeddings的值，即调整每个词的词向量?如何调整的？？？？？？？？？？？？？,nce_loss函数内部对输入向量做了调整？
        batch_inputs, batch_labels = generate_batch(
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        # 每2000次循环，计算一个平均loss并显示出来
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        # 每10000次循环，计算一次验证单词和全部单词的相似度，并将每个验证单词最相近的8个单词显示出来
        if step % 10000 == 0:
            sim = similarity.eval()  # 验证数据集词和所有词相似度的计算结果
            # 遍历每个验证单词
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                # sim[i, :]是第i个词和所有词的相似度，argsort返回数组从小到大元素的索引值，
                # 因为sim取了负号，所以实际(-sim[i, :]).argsort()的结果是相似度从大到小排列，并取了相似度最高的top_k个词的编码，注意，最高的是他本身，要忽略
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                # 对每个验证单词选择最相似的top_k个
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]  # 根据词的编码拿到对应的词
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()


# Step 6: Visualize the embeddings.


# pylint: disable=missing-docstring
# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]  # low_dim_embs是降维到2维的单词的空间向量
        plt.scatter(x, y)  # 显示散点图(单词的位置)
        plt.annotate(label,  # 展示单词本身
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)  # 保存图片到本地文件


try:
    # pylint: disable=g-import-not-at-top
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # sklearn.manifold.TSNE 实现降维，这里直接将原始的 128 维的嵌入向量降到 2 维
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    plot_only = 500  # 这里只展示词频最高的500个单词的可视化结果
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))
    # 从可视化的结果可以看出，距离相近的单词在语义上具有很高的相似性，在训练Word2Vec模型时，为了获得比较好的结构，我们可以使用大规模的预料库，同时需要对参数进行调试，选取最合适的值
except ImportError as ex:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
    print(ex)
