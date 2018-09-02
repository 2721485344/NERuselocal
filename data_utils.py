# encoding = utf8
import re
import math
import codecs
import random

import numpy as np
import jieba
jieba.initialize()


def create_dico(item_list):  #[['以', '冠', '心', '病', '收', '住', '入', '院', '。'],  ['音', '。']]
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:  #['以', '冠', '心', '病', '收', '住', '入', '院', '。']
            if item not in dico:
                dico[item] = 1 # {'无': 1}
            else:
                dico[item] += 1
    return dico #{'无': 1, '长': 1, '期': 1, '0': 7, '年': 1, '月': 1, '日': 1, '出': 1, '院': 1, '记': 1, '录': 1, '患': 1, '者': 1, '姓': 1}



def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))#[('<PAD>', 10000001), ('<UNK>', 10000000), ('0', 335904), ('，', 243952)]
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)} #{0: '<PAD>', 1: '<UNK>', 2: '0', 3: '，', 4: '：', 5: '。', 6: '无',}
    #item_to_id = {v: k for k, v in id_to_item.items()} #{'<PAD>': 0, '<UNK>': 1, '0': 2, '，': 3, '：': 4, '。': 5, '无': 6,}
    #return item_to_id,id_to_item
    return dict(zip(id_to_item.values(),id_to_item.keys())), id_to_item


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    #{B、I、O}       B表示命名实体的开头词   I表示命名实体非开始的词  O表示非命名实体词
    #{B、I、E、O、S} B表示命名实体的开头词   I表示命名实体非开始的词  O表示非命名实体词    E表示命名实体结尾词  S表示单个词的命名实体
    for i, tag in enumerate(tags): 
        if tag == 'O':
            continue
        split = tag.split('-')# 'B-SYM' ['B', 'SYM']
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):#['B', 'SYM']
    """
    IOB -> IOBES
    """
    new_tags = []  #O 非命名实体词 O  b实体开头 'B-SYM'  i不是实体开头 'I-SYM'  E实体结尾 E-SYM  S表示单个实体S-SYM
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)   #如果是非命名实体
        elif tag.split('-')[0] == 'B': #是开头的操作
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':  # 判断是不是实体
                new_tags.append(tag)  #i=0执行 并且 后面字的一个标注是I
            else:
                new_tags.append(tag.replace('B-', 'S-')) #如果后面不是I (可能是B)  就把当前替换成 单个实体 S
        elif tag.split('-')[0] == 'I':  #不是开头的操作
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':    # 判断是不是实体 #i=0执行 并且 后面字的一个标注是I
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-')) #如果后面不是I(可能是B)  就把当前替换成E
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


def insert_singletons(words, singletons, p=0.5):
    """
    Replace singletons by the unknown word with a probability p.
    """
    new_words = []
    for word in words:
        if word in singletons and np.random.uniform() < p:
            new_words.append(0)
        else:
            new_words.append(word)
    return new_words


def get_seg_features(string):
    """
    Segment text with jieba
    features are represented in bies format
    s donates single word
    """
    #切词  jieba sbie 对应数字 0123  string '无长期000年00月00日出院记录患者姓名：闫XX性别：男年龄：00岁入院日期：0000年00月00日00时00分出院日期：0000年00月00日00时00分共住院00天。'
    seg_feature = []

    for word in jieba.cut(string):
        if len(word) == 1:  #'无'
            seg_feature.append(0)
        else:
            tmp = [2] * len(word)  #'长期' [2, 2]  sbie  '入院日期'[2,2,2,2]  [1,2,2,3]
            tmp[0] = 1
            tmp[-1] = 3
            seg_feature.extend(tmp)# tmp [1, 3]  seg_feature [0, 1, 3]
    return seg_feature#[0, 1, 3, 1, 2, 3, 0, 1, 3, 0, 1, 3, 0, 1, 3, 1, 3, 1, 3, 1, 3, 0, 0, 1, 3, 1, 3, 0, 0, 1, 3, 0, 1, 3, 0, 1, 3, 1, 3, 0, 1, 2, 2, 3, 0, 1, 3, 0, 1, 3, 0, 1, 3, 0, 1, 3, 0, 1, 3, 1, 3, 0, 1, 2, 2, 3, 0, 1, 3, 0, 1, 3, 0, 1, 3, 0, 1, 3, 1, 3, 1, 3, 1, 3, 0, 0]


def create_input(data):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    inputs = list()
    inputs.append(data['chars'])
    inputs.append(data["segs"])
    inputs.append(data['tags'])
    return inputs


def load_word2vec(emb_path, id_to_word, word_dim, old_weights):
    """
    Load word embedding from pre-trained file
    embedding size must match
    """
    #把字典中所有的字转化为向量，假设字在字向量文件中，那就用字向量文件中的值初始化向量，
    new_weights = old_weights
    print('Loading pretrained embeddings from {}...'.format(emb_path))
    pre_trained = {}
    emb_invalid = 0
    for i, line in enumerate(codecs.open(emb_path, 'r', 'utf-8')):
        line = line.rstrip().split()
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = np.array(
                [float(x) for x in line[1:]]
            ).astype(np.float32) 
        else:
            emb_invalid += 1
    if emb_invalid > 0:
        print('WARNING: %i invalid lines' % emb_invalid)
    c_found = 0
    c_lower = 0
    c_zeros = 0
    n_words = len(id_to_word)
    # Lookup table initialization
    for i in range(n_words):
        word = id_to_word[i]
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
            c_found += 1
        elif word.lower() in pre_trained:
            new_weights[i] = pre_trained[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in pre_trained:
            new_weights[i] = pre_trained[
                re.sub('\d', '0', word.lower())
            ]
            c_zeros += 1
    print('Loaded %i pretrained embeddings.' % len(pre_trained))
    print('%i / %i (%.4f%%) words have been initialized with '
          'pretrained embeddings.' % (
        c_found + c_lower + c_zeros, n_words,
        100. * (c_found + c_lower + c_zeros) / n_words)
    )
    print('%i found directly, %i after lowercasing, '
          '%i after lowercasing + zero.' % (
        c_found, c_lower, c_zeros
    ))
    return new_weights


def full_to_half(s):
    """
    Convert full-width character to half-width one 
    """
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        char = chr(num)
        n.append(char)
    return ''.join(n)


def cut_to_sentence(text):
    """
    Cut text to sentences 
    """
    sentence = []
    sentences = []
    len_p = len(text)
    pre_cut = False
    for idx, word in enumerate(text):
        sentence.append(word)
        cut = False
        if pre_cut:
            cut=True
            pre_cut=False
        if word in u"!?\n":
            cut = True
            if len_p > idx+1:
                if text[idx+1] in ".\"\'?!":
                    cut = False
                    pre_cut=True

        if cut:
            sentences.append(sentence)
            sentence = []
    if sentence:
        sentences.append("".join(list(sentence)))
    return sentences


def replace_html(s):
    s = s.replace('&quot;','"')
    s = s.replace('&amp;','&')
    s = s.replace('&lt;','<')
    s = s.replace('&gt;','>')
    s = s.replace('&nbsp;',' ')
    s = s.replace("&ldquo;", "")
    s = s.replace("&rdquo;", "")
    s = s.replace("&mdash;","")
    s = s.replace("\xa0", " ")
    return(s)


def input_from_line(line, char_to_id):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    line = full_to_half(line)
    line = replace_html(line)
    inputs = list()
    inputs.append([line])
    line.replace(" ", "$")
    inputs.append([[char_to_id[char] if char in char_to_id else char_to_id["<UNK>"]
                   for char in line]])
    inputs.append([get_seg_features(line)])
    inputs.append([[]])
    return inputs


class BatchManager(object):

    def __init__(self, data,  batch_size):
        # data[0] [['无', '长', '期', '0', '0', '0', '年', '0', '0', '月', '0', '0', '日', '出', '院', '记', '录', '患', '者', '姓', '名', '：', '赵', 'X', 'X', '性', '别', '：', '女', '年', '龄', '：', '0', '0', '岁', '入', '院', '日', '期', '：', '0', '0', '0', '0', '年', '0', '0', '月', '0', '0', '日', '0', '0', '时', '0', '0', '分', '出', '院', '日', '期', '：', '0', '0', '0', '0', '年', '0', '0', '月', '0', '0', '日', '0', '0', '时', '0', '0', '分', '共', '住', '院', '0', '天', '。'], [6, 305, 110, 2, 2, 2, 35, 2, 2, 55, 2, 2, 51, 30, 12, 138, 205, 39, 37, 204, 188, 4, 829, 78, 78, 10, 167, 4, 248, 35, 240, 4, 2, 2, 175, 48, 12, 51, 110, 4, 2, 2, 2, 2, 35, 2, 2, 55, 2, 2, 51, 2, 2, 43, 2, 2, 29, 30, 12, 51, 110, 4, 2, 2, 2, 2, 35, 2, 2, 55, 2, 2, 51, 2, 2, 43, 2, 2, 29, 343, 68, 12, 2, 122, 5], [0, 1, 3, 1, 2, 3, 0, 1, 3, 0, 1, 3, 0, 1, 3, 1, 3, 1, 3, 1, 3, 0, 0, 1, 3, 1, 3, 0, 0, 1, 3, 0, 1, 3, 0, 1, 3, 1, 3, 0, 1, 2, 2, 3, 0, 1, 3, 0, 1, 3, 0, 1, 3, 0, 1, 3, 0, 1, 3, 1, 3, 0, 1, 2, 2, 3, 0, 1, 3, 0, 1, 3, 0, 1, 3, 0, 1, 3, 1, 3, 1, 3, 0, 0, 0], [0, 43, 44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 16, 0, 0, 0, 0, 0, 0, 0]]
        # 100 
        #data[0] 是一句话所有内容
        #data[0][0]对应的词
        #data[0][1]对应的词的id
        #data[0][2]对应的词 分词信息
        #data[0][3]对应的词 标记
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):#7827总共的句子，分批处理
        num_batch = int(math.ceil(len(data) /batch_size))#int(math.ceil(101218/60)) =1687 验证集
        sorted_data = sorted(data, key=lambda x: len(x[0])) #sorted_data[0]  [['。'], [5], [0], [0]]  句子长度排序从小到大 可以看成噪音 x[0]=data[0]
       #data=[[['无', '长', '期'],[1,2,3],[1,2,3],[1,2,3]],[['无', '长'],[1,2],[1,2],[1,2]]]
       #sorted_data = sorted(data, key=lambda x: len(x[0]))
       #print(sorted_data) 
       #[[['无', '长'], [1, 2], [1, 2], [1, 2]], [['无', '长', '期'], [1, 2, 3], [1, 2, 3], [1, 2, 3]]]

        batch_data = list()
        for i in range(num_batch):#把最短的句子pading最长的句子
            if len(sorted_data[i*int(batch_size)][0])<5:  #sorted_data[i*int(batch_size)]排序的小于5的句子去掉
                continue  # 
            batch_data.append(self.pad_data(sorted_data[i*int(batch_size) : (i+1)*int(batch_size)]))#根据id获取60句话
            #return batch_data #[0]第一批 [0][0]句话 [0][0][0] 单个字
    #batch_data[0] [[['。', 0], ['。', 0], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['岈', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['鳘', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['鳘', '。'], ['音', '。'], ['音', '。'], ['岈', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['鳘', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['岈', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['岈', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['音', '。'], ['鳘', '。'], ['音', '。'], ['音', '。'], ['音', '。']], [[5, 0], [5, 0], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [1558, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [1741, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [1741, 5], [33, 5], [33, 5], [1558, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [1741, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [1558, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [1558, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [33, 5], [1741, 5], [33, 5], [33, 5], [33, 5]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]]

    @staticmethod
    def pad_data(data):
        strings = []#代表的是一句话
        chars = []#每个字代表的id
        segs = []#分词信息
        targets = []#标记的id
        max_length = max([len(sentence[0]) for sentence in data]) #相同批次维度相同  一批次最大长度
        for line in data:
            string, char, seg, target = line
            padding = [0] * (max_length - len(string))#最大长度-实际长度，*[0],比如[0]*5=[0,0,0,0,0]
            strings.append(string + padding)#字
            chars.append(char + padding)#字id
            segs.append(seg + padding)#字分词
            targets.append(target + padding)#字标注
        return [strings, chars, segs, targets]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)#在一个bathsize 随机取
        for idx in range(self.len_data):
            yield self.batch_data[idx]
