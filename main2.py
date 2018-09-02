# encoding=utf8

import codecs
import pickle
import itertools
from collections import OrderedDict
import os
import tensorflow as tf
import numpy as np
from model import Model
from loader import load_sentences, update_tag_scheme
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset
from utils import get_logger, make_path, clean, create_model, save_model
from utils import print_config, save_config, load_config, test_ner
from data_utils import load_word2vec, create_input, input_from_line, BatchManager
root_path=os.getcwd()+os.sep
flags = tf.app.flags
flags.DEFINE_boolean("clean",       False,      "clean train folder")#清除文件
flags.DEFINE_boolean("train",       True,      "Whether train the model")#训练模型
# configurations for the model
flags.DEFINE_integer("seg_dim",     20,         "Embedding size for segmentation, 0 if not used")#切词信息
flags.DEFINE_integer("char_dim",    100,        "Embedding size for characters")#字向量，每个字用100维来表示
flags.DEFINE_integer("lstm_dim",    100,        "Num of hidden units in LSTM, or num of filters in IDCNN")#lsTM 维度数，或者卷积核个数为100 个
flags.DEFINE_string("tag_schema",   "iobes",    "tagging schema iobes or iob")#iob (三种状态)转iobes(五种状态)begin,middle,end,others,single

# configurations for training 训练模型配置参数
flags.DEFINE_float("clip",          5,          "Gradient clip")#梯度
flags.DEFINE_float("dropout",       0.5,        "Dropout rate")#
flags.DEFINE_float("batch_size",    60,         "batch size")#
flags.DEFINE_float("lr",            0.001,      "Initial learning rate")#
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")#
flags.DEFINE_boolean("pre_emb",     True,       "Wither use pre-trained embedding")#
flags.DEFINE_boolean("zeros",       True,      "Wither replace digits with zero")#把句子中有数字部分的变成0
flags.DEFINE_boolean("lower",       False,       "Wither lower case")#把句子中含有英文字母转换成小写

flags.DEFINE_integer("max_epoch",   100,        "maximum training epochs")#最大迭代次数
flags.DEFINE_integer("steps_check", 100,        "steps per checkpoint")
flags.DEFINE_string("ckpt_path",    "ckpt",      "Path to save model")
flags.DEFINE_string("summary_path", "summary",      "Path to store summaries")
flags.DEFINE_string("log_file",     "train.log",    "File for log")#训练时的文件日志
flags.DEFINE_string("map_file",     "maps.pkl",     "file for maps")
flags.DEFINE_string("vocab_file",   "vocab.json",   "File for vocab")
flags.DEFINE_string("config_file",  "config_file",  "File for config")
flags.DEFINE_string("script",       "conlleval",    "evaluation script")
flags.DEFINE_string("result_path",  "result",       "Path for results")
flags.DEFINE_string("emb_file",     os.path.join(root_path+"data", "vec.txt"),  "Path for pre_trained embedding")
flags.DEFINE_string("train_file",   os.path.join(root_path+"data", "example.train"),  "Path for train data")
flags.DEFINE_string("dev_file",     os.path.join(root_path+"data", "example.dev"),    "Path for dev data")
flags.DEFINE_string("test_file",    os.path.join(root_path+"data", "example.test"),   "Path for test data")

flags.DEFINE_string("model_type", "idcnn", "Model type, can be idcnn or bilstm")#idcnn膨胀卷积，bilstm双向卷积
#flags.DEFINE_string("model_type", "bilstm", "Model type, can be idcnn or bilstm")

FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much" #梯度计算
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]


# config for the model
def config_model(char_to_id, tag_to_id):
    config = OrderedDict()
    config["model_type"] = FLAGS.model_type
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = FLAGS.char_dim
    config["num_tags"] = len(tag_to_id)
    config["seg_dim"] = FLAGS.seg_dim
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size

    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    return config

#评价函数
def evaluate(sess, model, name, data, id_to_tag, logger):
    logger.info("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag)
    eval_lines = test_ner(ner_results, FLAGS.result_path)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1


def train():
    # load data sets
    train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)#训练集 101218 句子
    dev_sentences = load_sentences(FLAGS.dev_file, FLAGS.lower, FLAGS.zeros)#验证集 7827句子
    test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)#测试集 16804句子

    # Use selected tagging scheme (IOB / IOBES)
    update_tag_scheme(train_sentences, FLAGS.tag_schema) #更新标注iob转换成iobes
    update_tag_scheme(test_sentences, FLAGS.tag_schema)#更新标注iob转换成iobes
    update_tag_scheme(dev_sentences, FLAGS.tag_schema)#更新标注iob转换成iobes
    # create maps if not exist
    if not os.path.isfile(FLAGS.map_file):#判断maps.pkl是否存在
        # create dictionary for word
        if FLAGS.pre_emb:#是否使用预先训练的模型(训练好的字向量)  测试集的数据不在训练集中
            dico_chars_train = char_mapping(train_sentences, FLAGS.lower)[0]#字频统计下来 dico_chars
            dico_chars, char_to_id, id_to_char = augment_with_pretrained(
                dico_chars_train.copy(),
                FLAGS.emb_file,
                list(itertools.chain.from_iterable(#拉平，变成一个list
                    [[w[0] for w in s] for s in test_sentences])#w[0] 是个字
                )
            )#每个字建个字典，每个词建个字典
        else:
            #每个字的id,标记的id
            _c, char_to_id, id_to_char = char_mapping(train_sentences, FLAGS.lower)

        # Create a dictionary and a mapping for tags 每个标记的id
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)#字频，排序，写入文件
        #with open('maps.txt','w',encoding='utf8') as f1:
            #f1.writelines(str(char_to_id)+" "+id_to_char+" "+str(tag_to_id)+" "+id_to_tag+'\n')
        with open(FLAGS.map_file, "wb") as f:#持久化下来
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
    else:
        with open(FLAGS.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    # prepare data, get a collection of list containing index
    train_data = prepare_dataset(#字词 数字特征化
        train_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    dev_data = prepare_dataset(
        dev_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), 0, len(test_data)))

    train_manager = BatchManager(train_data, FLAGS.batch_size)#训练集每次60个句子进行迭代
    dev_manager = BatchManager(dev_data, 100)
    test_manager = BatchManager(test_data, 100)
    # make path for store log and model if not exist
    make_path(FLAGS)#创建文件log,result,ckpt
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = config_model(char_to_id, tag_to_id)#字符对应的id,标签对应的id
        save_config(config, FLAGS.config_file)#每次的数据不一样都要生成一个config_file，
    make_path(FLAGS) #创建文件log,result,ckpt 模型中的文件

    log_path = os.path.join("log", FLAGS.log_file)#读取log路径
    logger = get_logger(log_path)#定义log日志的写入格式
    print_config(config, logger)#写入log日志

    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True#设置GPU自适应，用多少使用多少
    #tf_config.gpu_options.per_process_gpu_memory_fraction=True 设置GPU的使用率，占比
    steps_per_epoch = train_manager.len_data#总共分多少批，取多少次
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        #模型初始化结束
        logger.info("start training")
        loss = []
        # with tf.device("/gpu:0"):没有Gpu注释掉  卷积神经网络要求句子的长度一样，
        for i in range(100):#迭代多少次，每次把数据拿过来
                for batch in train_manager.iter_batch(shuffle=True):#随机的拿
                    step, batch_loss = model.run_step(sess, True, batch)
                    loss.append(batch_loss)
                    if step % FLAGS.steps_check == 0:
                        iteration = step // steps_per_epoch + 1
                        logger.info("iteration:{} step:{}/{}, "
                                    "NER loss:{:>9.6f}".format(
                            iteration, step%steps_per_epoch, steps_per_epoch, np.mean(loss)))
                        loss = []
    
               # best = evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)比上次模型好的话，就保存
                if i%7==0:
                    save_model(sess, model, FLAGS.ckpt_path, logger)
            #evaluate(sess, model, "test", test_manager, id_to_tag, logger)


def evaluate_line():
    config = load_config(FLAGS.config_file)#从文件config_file 中读取配置数据
    #{'model_type': 'idcnn', 'num_chars': 3538, 'char_dim': 100, 'num_tags': 51, 'seg_dim': 20, 'lstm_dim': 100, 'batch_size': 20, 'emb_file': 'E:\\pythonWork3.6.2\\NERuselocal\\NERuselocal\\data\\vec.txt', 'clip': 5, 'dropout_keep': 0.5, 'optimizer': 'adam', 'lr': 0.001, 'tag_schema': 'iobes', 'pre_emb': True, 'zeros': True, 'lower': False}
    logger = get_logger(FLAGS.log_file)#写日志文件名字为 train.log
    # limit GPU memory
    tf_config = tf.ConfigProto()#实例化一个设置GPU的对象  函数用在创建session的时候，用来对session进行参数配置
    tf_config.gpu_options.allow_growth = True  #1动态申请显存需要多少使用多少 2限制GPU使用率 config.gpu_options.per_process_gpu_memory_fraction = 0.4
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        while True:
            # try:       
            #     line = input("请输入测试句子:")
            #     result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
            #     print(result)
            # except Exception as e:
            #     logger.info(e)

                line = input("请输入测试句子:")
                result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
                print(result)


def main(_):

    if 1:
        if FLAGS.clean:
            clean(FLAGS)
        train()
    else:
        evaluate_line()


if __name__ == "__main__":
    tf.app.run(main)



