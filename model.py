# encoding = utf-8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers

from utils import result_to_json
from data_utils import create_input, iobes_iob,iob_iobes


class Model(object):
    #初始化模型参数
    def __init__(self, config):

        self.config = config 
        #{'model_type': 'idcnn', 'num_chars': 3538, 'char_dim': 100, 'num_tags': 51, 'seg_dim': 20, 'lstm_dim': 100, 'batch_size': 20, 'emb_file': 'E:\\pythonWork3.6.2\\NERuselocal\\NERuselocal\\data\\vec.txt', 'clip': 5, 'dropout_keep': 0.5, 'optimizer': 'adam', 'lr': 0.001, 'tag_schema': 'iobes', 'pre_emb': True, 'zeros': True, 'lower': False}

        self.lr = config["lr"]# 0.001
        self.char_dim = config["char_dim"]#100 字的id
        self.lstm_dim = config["lstm_dim"]#100字的向量
        self.seg_dim = config["seg_dim"]#20 切词的id

        self.num_tags = config["num_tags"]#51
        self.num_chars = config["num_chars"]#样本中总字数 3538
        self.num_segs = 4
        #global_step全局 trainable 不需要梯度更新 做了F1评价
        self.global_step = tf.Variable(0, trainable=False) #<tf.Variable 'Variable:0' shape=() dtype=int32_ref>
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)#创建一个变量，框架，
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        #正态分布的方法
        self.initializer = initializers.xavier_initializer() #<function variance_scaling_initializer.<locals>._initializer at 0x000000882230E400>




        # add placeholders for the model
        #把每个字放进来。[size,leng]
        self.char_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None], 
                                          name="ChatInputs") #<tf.Tensor 'ChatInputs:0' shape=(?, ?) dtype=int32>
        self.seg_inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name="SegInputs")

        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None],
                                      name="Targets")
        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32,
                                      name="Dropout") #<tf.Tensor 'Dropout:0' shape=<unknown> dtype=float32>
        #sign 指数函数分段函数 char_inputs转换成01 
        used = tf.sign(tf.abs(self.char_inputs)) #<tf.Tensor 'Sign:0' shape=(?, ?) dtype=int32>
        #reduce_sum求和 按reduction_indices 维度求和
        length = tf.reduce_sum(used, reduction_indices=1) #<tf.Tensor 'Sum:0' shape=(?,) dtype=int32>
        #length转换成浮点32位的
        self.lengths = tf.cast(length, tf.int32) #<tf.Tensor 'Sum:0' shape=(?,) dtype=int32>
        #分多少步
        self.batch_size = tf.shape(self.char_inputs)[0]#<tf.Tensor 'strided_slice:0' shape=() dtype=int32>
        #一句话的长度
        self.num_steps = tf.shape(self.char_inputs)[-1] #<tf.Tensor 'strided_slice_1:0' shape=() dtype=int32>


        #Add model type by crownpku bilstm or idcnn
        self.model_type = config['model_type'] #'idcnn' 选择模型的类型
        #parameters for idcnn
        self.layers = [#膨胀卷积网络的参数
            {
                'dilation': 1
                },
            {
                'dilation': 1
                },
            {
                'dilation': 2
                },
        ]
        self.filter_width = 3 #卷积和的大小
        self.num_filter = self.lstm_dim  #100 双向lstm 是卷积核的个数
        #char_dim 100字的维度数和20位是切词的分词的维度数   embedding_dim嵌入的维度
        self.embedding_dim = self.char_dim + self.seg_dim #120
        self.repeat_times = 4 #像goognet一样的几个分支
        self.cnn_output_width = 0#输出的宽度为0

        # embeddings for chinese character and segmentation representation
        #把特征嵌入进来  卷积
        embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, config)
        #<tf.Tensor 'char_embedding/concat:0' shape=(?, ?, 120) dtype=float32>

        if self.model_type == 'bilstm':#双向lstm
            # apply dropout before feed to lstm layer  embedding dropout下
            model_inputs = tf.nn.dropout(embedding, self.dropout)

            # bi-directional lstm layer  lstm_dim内部维度数，解码的长度lengths
            model_outputs = self.biLSTM_layer(model_inputs, self.lstm_dim, self.lengths)

            # logits for tags  logits预测的直，预测标签
            self.logits = self.project_layer_bilstm(model_outputs)

        elif self.model_type == 'idcnn':#膨胀卷积网络
            # apply dropout before feed to idcnn layer
            model_inputs = tf.nn.dropout(embedding, self.dropout)

            # ldcnn layer  卷积层进行卷积 输入数据到特征的抽取 与标签一样
            model_outputs = self.IDCNN_layer(model_inputs)

            # logits for tags  model_outputs特征，乘以权重w加上b
            #线性映射logits和model_output输出维度不一样，所以要做线性映射 
            self.logits = self.project_layer_idcnn(model_outputs) #model_outputs这个特征做全连接最终的特征

        else:
            raise KeyError

        # loss of the model 特征，和 维度来计算损失  之后优化参数
        self.loss = self.loss_layer(self.logits, self.lengths)
        #<tf.Tensor 'crf_loss/Mean:0' shape=() dtype=float32>
        with tf.variable_scope("optimizer"):#定义优化器
            optimizer = self.config["optimizer"]#adam
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError

            # apply grad clip to avoid gradient explosion  
            #计算梯度
            grads_vars = self.opt.compute_gradients(self.loss) #len(grads_vars) 12
            #[(<tensorflow.python.framework.ops.IndexedSlices object at 0x00000088298F6400>,
            #<tf.Variable 'char_embedding/char_embedding:0' shape=(3538, 100) dtype=float32_ref>),
            
            #(<tensorflow.python.framework.ops.IndexedSlices object at 0x000000882991FBA8>, 
            #<tf.Variable 'char_embedding/seg_embedding/seg_embedding:0' shape=(4, 20) dtype=float32_ref>), 
            
            #(<tf.Tensor 'optimizer/gradients/idcnn/init_layer_grad/tuple/control_dependency_1:0' shape=(1, 3, 120, 100) dtype=float32>,
            #<tf.Variable 'idcnn/idcnn_filter:0' shape=(1, 3, 120, 100) dtype=float32_ref>), 
            
            #(<tf.Tensor 'optimizer/gradients/AddN_13:0' shape=(1, 3, 100, 100) dtype=float32>, 
            #<tf.Variable 'idcnn/atrous-conv-layer-0/filterW:0' shape=(1, 3, 100, 100) dtype=float32_ref>), 
            
            #(<tf.Tensor 'optimizer/gradients/AddN_12:0' shape=(100,) dtype=float32>,
            #<tf.Variable 'idcnn/atrous-conv-layer-0/filterB:0' shape=(100,) dtype=float32_ref>),
            
            #(<tf.Tensor 'optimizer/gradients/AddN_11:0' shape=(1, 3, 100, 100) dtype=float32>, 
            #<tf.Variable 'idcnn/atrous-conv-layer-1/filterW:0' shape=(1, 3, 100, 100) dtype=float32_ref>),
            
            #(<tf.Tensor 'optimizer/gradients/AddN_10:0' shape=(100,) dtype=float32>,
            #<tf.Variable 'idcnn/atrous-conv-layer-1/filterB:0' shape=(100,) dtype=float32_ref>), 
            
            #(<tf.Tensor 'optimizer/gradients/AddN_9:0' shape=(1, 3, 100, 100) dtype=float32>,
            #<tf.Variable 'idcnn/atrous-conv-layer-2/filterW:0' shape=(1, 3, 100, 100) dtype=float32_ref>),
            
            #(<tf.Tensor 'optimizer/gradients/AddN_8:0' shape=(100,) dtype=float32>, 
            #<tf.Variable 'idcnn/atrous-conv-layer-2/filterB:0' shape=(100,) dtype=float32_ref>), 
            
            #(<tf.Tensor 'optimizer/gradients/project/logits/xw_plus_b/MatMul_grad/tuple/control_dependency_1:0' shape=(400, 51) dtype=float32>,
            #<tf.Variable 'project/logits/W:0' shape=(400, 51) dtype=float32_ref>), 
            
            #(<tf.Tensor 'optimizer/gradients/project/logits/xw_plus_b_grad/tuple/control_dependency_1:0' shape=(51,) dtype=float32>, 
            #<tf.Variable 'project/logits/b:0' shape=(51,) dtype=float32_ref>), 
            
            #(<tf.Tensor 'optimizer/gradients/AddN_4:0' shape=(52, 52) dtype=float32>,
            #<tf.Variable 'crf_loss/transitions:0' shape=(52, 52) dtype=float32_ref>)]
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in grads_vars]  #梯度进行截断（更新）
            #[[<tf.Tensor 'optimizer/clip_by_value:0' shape=(?, 100) dtype=float32>, 
            #<tf.Variable 'char_embedding/char_embedding:0' shape=(3538, 100) dtype=float32_ref>],
            
            #[<tf.Tensor 'optimizer/clip_by_value_1:0' shape=(?, 20) dtype=float32>, 
            #<tf.Variable 'char_embedding/seg_embedding/seg_embedding:0' shape=(4, 20) dtype=float32_ref>], 
            
            #[<tf.Tensor 'optimizer/clip_by_value_2:0' shape=(1, 3, 120, 100) dtype=float32>, 
            #<tf.Variable 'idcnn/idcnn_filter:0' shape=(1, 3, 120, 100) dtype=float32_ref>], 
            
            #[<tf.Tensor 'optimizer/clip_by_value_3:0' shape=(1, 3, 100, 100) dtype=float32>, 
            #<tf.Variable 'idcnn/atrous-conv-layer-0/filterW:0' shape=(1, 3, 100, 100) dtype=float32_ref>],
            
            #[<tf.Tensor 'optimizer/clip_by_value_4:0' shape=(100,) dtype=float32>, 
            #<tf.Variable 'idcnn/atrous-conv-layer-0/filterB:0' shape=(100,) dtype=float32_ref>],
            
            #[<tf.Tensor 'optimizer/clip_by_value_5:0' shape=(1, 3, 100, 100) dtype=float32>,
            #<tf.Variable 'idcnn/atrous-conv-layer-1/filterW:0' shape=(1, 3, 100, 100) dtype=float32_ref>],
            
            #[<tf.Tensor 'optimizer/clip_by_value_6:0' shape=(100,) dtype=float32>, 
            #<tf.Variable 'idcnn/atrous-conv-layer-1/filterB:0' shape=(100,) dtype=float32_ref>], 
            
            #[<tf.Tensor 'optimizer/clip_by_value_7:0' shape=(1, 3, 100, 100) dtype=float32>, 
            #<tf.Variable 'idcnn/atrous-conv-layer-2/filterW:0' shape=(1, 3, 100, 100) dtype=float32_ref>], 
            
            #[<tf.Tensor 'optimizer/clip_by_value_8:0' shape=(100,) dtype=float32>,
            #<tf.Variable 'idcnn/atrous-conv-layer-2/filterB:0' shape=(100,) dtype=float32_ref>], 
            
            #[<tf.Tensor 'optimizer/clip_by_value_9:0' shape=(400, 51) dtype=float32>,
            #<tf.Variable 'project/logits/W:0' shape=(400, 51) dtype=float32_ref>], 
            
            #[<tf.Tensor 'optimizer/clip_by_value_10:0' shape=(51,) dtype=float32>, 
            #<tf.Variable 'project/logits/b:0' shape=(51,) dtype=float32_ref>], 
            
            #[<tf.Tensor 'optimizer/clip_by_value_11:0' shape=(52, 52) dtype=float32>, 
            #<tf.Variable 'crf_loss/transitions:0' shape=(52, 52) dtype=float32_ref>]]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)#global_step要求解的一个值
            #更新梯度
        # saver of the model 模型保存
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def embedding_layer(self, char_inputs, seg_inputs, config, name=None):
        """
        :param char_inputs: one-hot encoding of sentence  输入数据
        :param seg_inputs: segmentation feature  嵌入的分词的信息
        :param config: wither use segmentation feature  参数配置
        :return: [1, num_steps, embedding size], 
        """
        #高:3 血:22 糖:23 和:24 高:3 血:22 压:25 char_inputs=[3,22,23,24,3,22,25]
        #高血糖 和 高血压 seg_inputs 高血糖=[1,2,3] 和=[0] 高血压=[1,2,3]  seg_inputs=[1,2,3,0,1,2,3]
        embedding = []
        self.char_inputs_test=char_inputs #<tf.Tensor 'ChatInputs:0' shape=(?, ?) dtype=int32>
        self.seg_inputs_test=seg_inputs #<tf.Tensor 'SegInputs:0' shape=(?, ?) dtype=int32>
        with tf.variable_scope("char_embedding" if not name else name), tf.device('/gpu:0'):
            self.char_lookup = tf.get_variable(
                name="char_embedding",
                    shape=[self.num_chars, self.char_dim],
                    initializer=self.initializer) #初始化字向量，把字的id转换成100维的向量
            #创建一个表get_variable 二维的 num_chars行---字的总数  列指定字的维度数
            #<tf.Variable 'char_embedding/char_embedding:0' shape=(3538, 100) dtype=float32_ref>
            #输入char_inputs='常' 对应的字典的索引/编号/value为：8
            #self.char_lookup=[2677*100]的向量，char_inputs字对应在字典的索引/编号/key=[1]
            #tf.nn.embedding_lookup（tensor是一个对角矩阵, id是一个一维的tensor）:tensor就是输入张量，id就是张量对应的索引
            #char_inputs每个字转化成100维的字向量（句子的总数，每个句子的长度）每个字要用100维度的数据来表示
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            #self.embedding1.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            if config["seg_dim"]:#20
                with tf.variable_scope("seg_embedding"), tf.device('/gpu:0'):
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        #shape=[4*20]
                        shape=[self.num_segs, self.seg_dim],#num_segs分词的状态信息seg_dim字对应的id
                        initializer=self.initializer)#initializer正态分布，随机初始化参数
                    #每个字都有四种状态
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
            embed = tf.concat(embedding, axis=-1)#按最后一个维度拼接 bichsize 句子中字的个数
        self.embed_test=embed #<tf.Tensor 'char_embedding/concat:0' shape=(?, ?, 120) dtype=float32>
        self.embedding_test=embedding
        # [<tf.Tensor 'char_embedding/embedding_lookup:0' shape=(?, ?, 100) dtype=float32>, 
        #<tf.Tensor 'char_embedding/seg_embedding/embedding_lookup:0' shape=(?, ?, 20) dtype=float32>]
        return embed


    #IDCNN layer 
    def IDCNN_layer(self, model_inputs, 
                    name=None):
        """
        :param idcnn_inputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, cnn_output_width]
        """
        #tf.expand_dims会向tensor中插入一个维度，插入位置就是参数代表的位置（维度从0开始）。一般4维
        model_inputs = tf.expand_dims(model_inputs, 1)#<tf.Tensor 'ExpandDims:0' shape=(?, 1, ?, 120) dtype=float32>
        self.model_inputs_test=model_inputs
        reuse = False
        if self.dropout == 1.0:
            reuse = True
        with tf.variable_scope("idcnn" if not name else name):
            #shape=[1*3*120*100][高，宽，输入，输出]
            shape=[1, self.filter_width, self.embedding_dim,
                   self.num_filter]
            print(shape)#[1, 3, 120, 100]
            #初始化一个权重
            filter_weights = tf.get_variable(
                "idcnn_filter",
                shape=[1, self.filter_width, self.embedding_dim,
                       self.num_filter],
                initializer=self.initializer)
            #<tf.Variable 'idcnn/idcnn_filter:0' shape=(1, 3, 120, 100) dtype=float32_ref>

            """
            shape of input = [batch, in_height, in_width, in_channels]
            shape of filter = [filter_height, filter_width, in_channels, out_channels]
            """ 
            #conv1d 比 conv2d 少个维度  model_inputs模型的输入 filter_weights#1*3 的卷积和 卷积的参数
            layerInput = tf.nn.conv2d(model_inputs,
                                      filter_weights,
                                      strides=[1, 1, 1, 1],
                                      padding="SAME",
                                      name="init_layer",use_cudnn_on_gpu=True)
            #<tf.Tensor 'idcnn/init_layer:0' shape=(?, 1, ?, 100) dtype=float32>(100-3+2*1)/1+1
            self.layerInput_test=layerInput
            finalOutFromLayers = []

            totalWidthForLastDim = 0
            for j in range(self.repeat_times):#4 分四个分支进行重复的训练
                #   卷三次 self.layers [{'dilation': 1}, {'dilation': 1}, {'dilation': 2}]
                for i in range(len(self.layers)):
                    #1,1,2 膨胀系数dilation 是1相当于没有膨胀 padding="SAME",是一样的。
                    dilation = self.layers[i]['dilation']
                    isLast = True if i == (len(self.layers) - 1) else False #当i等于2的时候是最后一次卷积
                    with tf.variable_scope("atrous-conv-layer-%d" % i,
                                           reuse=True#  在variable_scope作用域下都是True 也就是每次w,b,都会更新，用的是同一个变量
                                           #控制dropout True 1 Fase 0.5
                                           if (reuse or j > 0) else False): #节省空间
                        #这个命名空间下有那个key,value 两个key "atrous-conv-layer-%d" % i reuse
                        #w 卷积核的高度，卷积核的宽度，图像通道数，卷积核个数
                        w = tf.get_variable(
                            "filterW",
                            shape=[1, self.filter_width, self.num_filter,
                                   self.num_filter],
                            initializer=tf.contrib.layers.xavier_initializer())#初始化卷积和
                        if j==1 and i==1:
                            self.w_test_1=w
                        if j==2 and i==1:
                            self.w_test_2=w                            
                        b = tf.get_variable("filterB", shape=[self.num_filter])#初始化偏值
#tf.nn.atrous_conv2d(value,filters,rate,padding,name=None）
    #除去name参数用以指定该操作的name，与方法有关的一共四个参数：                  
    #value： 
    #指需要做卷积的输入图像，要求是一个4维Tensor，具有[batch, height, width, channels]这样的shape，具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数] 
    #filters： 
    #相当于CNN中的卷积核，要求是一个4维Tensor，具有[filter_height, filter_width, channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，同理这里第三维channels，就是参数value的第四维
    #rate： 
    #要求是一个int型的正数，正常的卷积操作应该会有stride（即卷积核的滑动步长），但是空洞卷积是没有stride参数的，
    #这一点尤其要注意。取而代之，它使用了新的rate参数，那么rate参数有什么用呢？它定义为我们在输入
    #图像上卷积时的采样间隔，你可以理解为卷积核当中穿插了（rate-1）数量的“0”，
    #把原来的卷积核插出了很多“洞洞”，这样做卷积时就相当于对原图像的采样间隔变大了。
    #具体怎么插得，可以看后面更加详细的描述。此时我们很容易得出rate=1时，就没有0插入，
    #此时这个函数就变成了普通卷积。  
    #padding： 
    #string类型的量，只能是”SAME”,”VALID”其中之一，这个值决定了不同边缘填充方式。
    #ok，完了，到这就没有参数了，或许有的小伙伴会问那“stride”参数呢。其实这个函数已经默认了stride=1，也就是滑动步长无法改变，固定为1。
    #结果返回一个Tensor，填充方式为“VALID”时，返回[batch,height-2*(filter_width-1),width-2*(filter_height-1),out_channels]的Tensor，填充方式为“SAME”时，返回[batch, height, width, out_channels]的Tensor，这个结果怎么得出来的？先不急，我们通过一段程序形象的演示一下空洞卷积。                        
                        conv = tf.nn.atrous_conv2d(layerInput,
                                                   w,
                                                   rate=dilation,
                                                   padding="SAME")#膨胀卷积
                        self.conv_test=conv 
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv)
                        if isLast:
                            finalOutFromLayers.append(conv)
                            totalWidthForLastDim += self.num_filter
                        layerInput = conv
            finalOut = tf.concat(axis=3, values=finalOutFromLayers) #4层信息都放进来
            #<tf.Tensor 'idcnn/concat:0' shape=(?, ?, ?, 400) dtype=float32>
            keepProb = 1.0 if reuse else 0.5
            finalOut = tf.nn.dropout(finalOut, keepProb)
            #<tf.Tensor 'idcnn/dropout/mul:0' shape=(?, ?, ?, 400) dtype=float32>
            #Removes dimensions of size 1 from the shape of a tensor. 
                #从tensor中删除所有大小是1的维度

                #Given a tensor input, this operation returns a tensor of the same type with all dimensions of size 1 removed. If you don’t want to remove all size 1 dimensions, you can remove specific size 1 dimensions by specifying squeeze_dims. 

                #给定张量输入，此操作返回相同类型的张量，并删除所有尺寸为1的尺寸。 如果不想删除所有尺寸1尺寸，可以通过指定squeeze_dims来删除特定尺寸1尺寸。
            finalOut = tf.squeeze(finalOut, [1])#把添加的维度去掉 宽，高不变
            #<tf.Tensor 'idcnn/Squeeze:0' shape=(?, ?, 400) dtype=float32>
            finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim]) 
            #把前两个维度乘起来，最后面的维度不变
            #<tf.Tensor 'idcnn/Reshape:0' shape=(?, 400) dtype=float32>
            self.cnn_output_width = totalWidthForLastDim #400
            return finalOut#特征提取结束

    def project_layer_bilstm(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    #Project layer for idcnn by crownpku
    #Delete the hidden layer, and change bias initializer
    def project_layer_idcnn(self, idcnn_outputs, name=None):
        """
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        #做的是*w+b 线性映射
        with tf.variable_scope("project"  if not name else name):

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.cnn_output_width, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b",  initializer=tf.constant(0.001, shape=[self.num_tags]))

                pred = tf.nn.xw_plus_b(idcnn_outputs, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags]) #num_steps句子的个数 num_tags标签个数

    def loss_layer(self, project_logits, lengths, name=None):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags] num_steps字的个数
        :return: scalar loss
        """
        #最终的特征采用条件随机解码器  
        #条件随机 看上下文 预测下个字的tag 看特征函数，上个字，状态转移，上个解码tag是啥概率是多大
        #SoftMax不看上下文
        #来算 project_logits 特征  lengths句子的解码长度
        with tf.variable_scope("crf_loss"  if not name else name):
            small = -1000.0#定义看状态，看特征
            # pad logits for crf loss
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)#batch_size*1*(num_tags+1)初始化
            #初始化一些1
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            #project_logits 拼接 pad_logits
            logits = tf.concat([project_logits, pad_logits], axis=-1)#最后一维concat project_logits=51+1   52
            #<tf.Tensor 'crf_loss/concat_1:0' shape=(?, ?, 52) dtype=float32>
            logits = tf.concat([start_logits, logits], axis=1)
            #<tf.Tensor 'crf_loss/concat_2:0' shape=(?, ?, 52) dtype=float32>
            targets = tf.concat(
                [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

            self.trans = tf.get_variable(#初始化一个状态转移矩阵，也就是状态的转移
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)#特征转移矩阵，状态转移矩阵
            #crf_log_likelihood在一个条件随机场里面计算标签序列的log-likelihood
            #inputs: 一个形状为[batch_size, max_seq_len, num_tags] 的tensor,
            #一般使用BILSTM处理之后输出转换为他要求的形状作为CRF层的输入. 
            #tag_indices: 一个形状为[batch_size, max_seq_len] 的矩阵,其实就是真实标签. 
            #sequence_lengths: 一个形状为 [batch_size] 的向量,表示每个序列的长度. 
            #transition_params: 形状为[num_tags, num_tags] 的转移矩阵    
            #log_likelihood: 标量,log-likelihood 
            #transition_params: 形状为[num_tags, num_tags] 的转移矩阵               
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,#转移矩阵
                sequence_lengths=lengths+1)
            return tf.reduce_mean(-log_likelihood)

    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data 
        :return: structured data to feed
        """
        _, chars, segs, tags = batch#_六十个句子 chars每个字id化 segs 分词id  tag 标签
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.seg_inputs: np.asarray(segs),
            self.dropout: 1.0,
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch 是否训练模式
        :param batch: a dict containing batch data 数据
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)#做的字典输入数据
        if is_train:
            global_step, loss,_,char_lookup_out,seg_lookup_out,char_inputs_test,seg_inputs_test,embed_test,embedding_test,\
                model_inputs_test,layerInput_test,conv_test,w_test_1,w_test_2,char_inputs_test= sess.run(
                    [self.global_step, self.loss, self.train_op,self.char_lookup,self.seg_lookup,self.char_inputs_test,self.seg_inputs_test,\
                 self.embed_test,self.embedding_test,self.model_inputs_test,self.layerInput_test,self.conv_test,self.w_test_1,self.w_test_2,self.char_inputs],
                feed_dict)
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits
    #解码
    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        small = -1000.0
        start = np.asarray([[small]*self.num_tags +[0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)

            paths.append(path[1:])
        return paths

    def evaluate(self, sess, data_manager, id_to_tag):
        """
        :param sess: session  to run the model 
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = self.trans.eval()
        for batch in data_manager.iter_batch():
            strings = batch[0]
            tags = batch[-1]
            lengths, scores = self.run_step(sess, False, batch)
            batch_paths = self.decode(scores, lengths, trans)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
                #gold = iob_iobes([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                #pred = iob_iobes([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])                
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results

    def evaluate_line(self, sess, inputs, id_to_tag):
        trans = self.trans.eval(session=sess)
        lengths, scores = self.run_step(sess, False, inputs)
        batch_paths = self.decode(scores, lengths, trans)
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        return result_to_json(inputs[0][0], tags)
