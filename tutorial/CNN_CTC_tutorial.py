# coding: utf-8

# # 利用thchs30为例建立一个语音识别系统
# 
# 
# - 数据处理
# - 搭建模型
#     - DFCNN
# 
# 论文地址：http://www.infocomm-journal.com/dxkx/CN/article/downloadArticleFile.do?attachType=PDF&id=166970

# In[ ]:


#!wget http://www.openslr.org/resources/18/data_thchs30.tgz


# In[ ]:


#!tar zxvf data_thchs30.tgz


# ## 1. 特征提取
# 
# input为输入音频数据，需要转化为频谱图数据，然后通过cnn处理图片的能力进行识别。
# 
# **1. 读取音频文件**

# In[5]:


import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os

# 随意搞个音频做实验
filepath = 'test.wav'
filepath = '1A - 128 - Bounce Intro(djoffice).mp3'#G:\\舞曲\\舞曲\\
filepath = '1A - 128 - Captain Jack - Captain Jack (Oski & Citos Bootleg)(djoffice).wav'#G:\\舞曲\\舞曲\\

fs, wavsignal = wav.read(filepath)
print("采样率: %d" % fs)#sample_rate  采样率: 16000
print(wavsignal)
if wavsignal.dtype == np.int16:#PCM16位整形
    print("PCM16位整形")
if wavsignal.dtype == np.float32:
    print("PCM32位浮点")
    
# plt.plot(wavsignal)
# plt.show()


# **2. 构造汉明窗**
# In[6]:
x=np.linspace(0, 400 - 1, 400, dtype = np.int64)
w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1)) #0.5 - 0.46*cos(2N*X/399)
# plt.plot(w)
# plt.show()


# **3. 对数据分帧**
# 
# - 帧长： 25ms
# - 帧移： 10ms
# 
# 
# ```
# 采样点（s） = fs
# 采样点（ms）= fs / 1000  #  采样点数量/毫秒
# 采样点（帧）= fs / 1000 * 帧长  #400
# ```
# In[8]:
time_window = 25
window_length = fs // 1000 * time_window  #400


# **4. 分帧加窗**

# In[9]:
# 分帧  #25MS一帧  ,400（25*16）个数据
p_begin = 0
p_end = p_begin + window_length
frame = wavsignal[p_begin:p_end]
# plt.plot(frame)
# plt.show()
# 加窗 乘以W怎么从郑玄变成雨轩了？
frame = frame * w  #25MS的一帧数据
# plt.plot(frame)
# plt.show()


# **5. 傅里叶变换**
# 
# 所谓时频图就是将时域信息转换到频域上去，具体原理可百度。人耳感知声音是通过

# In[10]:
from scipy.fftpack import fft

# 进行快速傅里叶变换
a = fft(frame) #SHAPE不变
frame_fft = np.abs(fft(frame))[:200]# 400---》200？  只取前面200 ？
# plt.plot(frame_fft)
# plt.show()

# 取对数，求db
frame_log = np.log(frame_fft)
# plt.plot(frame_log)
# plt.show()


# - 分帧
# - 加窗
# - 傅里叶变换

# In[11]:
import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import fft


# 获取信号的时频图
def compute_fbank(file):
	x=np.linspace(0, 400 - 1, 400, dtype = np.int64)
	w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1) ) # 汉明窗
	fs, wavsignal = wav.read(file)
	# wav波形 加时间窗以及时移10ms
	time_window = 25 # 单位ms
	window_length = fs / 1000 * time_window # 计算窗长度的公式，目前全部为400固定值
	wav_arr = np.array(wavsignal)
	wav_length = len(wavsignal)
	range0_end = int(len(wavsignal)/fs*1000 - time_window) // 10 # 计算循环终止的位置，也就是最终生成的窗数
	data_input = np.zeros((range0_end, 200), dtype = np.float) # 用于存放最终的频率特征数据，一行一个窗，一列一个特征
	data_line = np.zeros((1, 400), dtype = np.float)
	for i in range(0, range0_end):
		p_start = i * 160 #窗口移动10MS
		p_end = p_start + 400  #窗口长度25毫秒
		data_line = wav_arr[p_start:p_end]
		data_line = data_line * w # 加窗
		data_line = np.abs(fft(data_line))
		data_input[i]=data_line[0:200] # 设置为400除以2的值（即200）是取一半数据，因为是对称的。最终数据存入data_input
	data_input = np.log(data_input + 1)
	#data_input = data_input[::]
	return data_input
# 
# - 该函数提取音频文件的时频图



# In[12]:
import matplotlib.pyplot as plt
filepath = 'test.wav'
# filepath = '1A - 128 - Bounce Intro(djoffice).mp3'#G:\\舞曲\\舞曲\\
# filepath = '1A - 128 - Captain Jack - Captain Jack (Oski & Citos Bootleg)(djoffice).wav'


# a = compute_fbank(filepath)
# plt.imshow(a.T, origin = 'lower')# 转置过了，所以图片反过来了
# plt.show()


# ## 2. 数据处理
# 
# #### 下载数据
# thchs30: http://www.openslr.org/18/
# 
# ### 2.1 生成音频文件和标签文件列表
# 考虑神经网络训练过程中接收的输入输出。首先需要batch_size内数据需要统一数据的shape。
# 
# **格式为**：[batch_size, time_step, feature_dim]
# 
# 然而读取的每一个sample的时间轴长都不一样，所以需要对时间轴进行处理，选择batch内最长的那个时间为基准，进行padding。这样一个batch内的数据都相同，
# 就能进行并行训练啦。
# 

# In[13]:
#E:\work\github_code\DeepSpeechRecognition\data
source_file = "./dataset/sound/data_thchs30"
# #### 定义函数`source_get`，获取音频文件(.wav)及标注文件(.trn)列表
# 
# 形如：
# ```
# E:\Data\thchs30\data_thchs30\data\A11_0.wav.trn
# E:\Data\thchs30\data_thchs30\data\A11_1.wav.trn
# E:\Data\thchs30\data_thchs30\data\A11_10.wav.trn
# E:\Data\thchs30\data_thchs30\data\A11_100.wav.trn
# E:\Data\thchs30\data_thchs30\data\A11_102.wav.trn
# E:\Data\thchs30\data_thchs30\data\A11_103.wav.trn
# E:\Data\thchs30\data_thchs30\data\A11_104.wav.trn
# E:\Data\thchs30\data_thchs30\data\A11_105.wav.trn
# E:\Data\thchs30\data_thchs30\data\A11_106.wav.trn
# E:\Data\thchs30\data_thchs30\data\A11_107.wav.trn
# ```

# In[14]:

#返回波形文件和标签文件的列表
def source_get(source_file):
    train_file = source_file + '/data' #data包含了所有的数据列表
    label_lst = []
    wav_lst = []
    for root, dirs, files in os.walk(train_file):
        for file in files:
            if file.endswith('.wav') or file.endswith('.WAV'):
                wav_file = os.sep.join([root, file])
                label_file = wav_file + '.trn'#直接按照后缀名添加就可以了，不需要搜索，因为是WAV TRN文件是一一对应的。
                wav_lst.append(wav_file)
                label_lst.append(label_file)

    return label_lst, wav_lst

label_lst, wav_lst = source_get(source_file)

print(label_lst[:10])#打出来看看
print(wav_lst[:10])


# #### 确认相同id对应的音频文件和标签文件相同#只看前面10000个，肯定相同#  thchs30数据集就1万个训练数据  len(wav_lst)=13388

# In[ ]:
for i in range(len(wav_lst)):
    wavname = (wav_lst[i].split('/')[-1]).split('.')[0]
    labelname = (label_lst[i].split('/')[-1]).split('.')[0]
    if wavname != labelname:
        print('error')


# ### 2.2 label数据处理
# #### 定义函数`read_label`读取音频文件对应的拼音label

# In[13]:


def read_label(label_file):#读出TRN标签文件中拼音那一行
    with open(label_file, 'r', encoding='utf8') as f:
        data = f.readlines()
        return data[1]#第二行是拼音

print(read_label(label_lst[0]))

def gen_label_data(label_lst):
    label_data = []
    for label_file in label_lst:
        pny = read_label(label_file)
        label_data.append(pny.strip('\n'))
    return label_data

label_data = gen_label_data(label_lst)#读所有TRN文件列表里面的拼音
print(len(label_data))#13388行  = 训练 验证 测试


# #### 为label建立拼音到id的映射，即词典
# In[14]:
def mk_vocab(label_data):
    vocab = []
    for line in label_data:
        line = line.split(' ')
        for pny in line:
            if pny not in vocab:#去除重复拼音
                vocab.append(pny)
    vocab.append('_')
    return vocab

vocab = mk_vocab(label_data)
print(len(vocab))#根据数据集生成的1209个拼音单词的词典， DICT.TXT里面是1424个，2者应该是一个意思？


# #### 有了词典就能将读取到的label映射到对应的id，返回拼音在词典中的索引
# In[15]:
def word2id(line, vocab):
    return [vocab.index(pny) for pny in line.split(' ')]

label_id = word2id(label_data[0], vocab)
print(label_data[0])
print(label_id)#算出第一个WAV文件中拼音的序号或索引

label_id = word2id(label_data[100], vocab)
print(label_data[100])
print(label_id)


# #### 总结:
# 我们提取出了每个音频文件对应的拼音标签`label_data`，通过索引就可以获得该索引的标签。
# 
# 也生成了对应的拼音词典.由此词典，我们可以映射拼音标签为id序列。
# 
# 输出：
# - vocab
# - label_data

# In[16]:
print(vocab[:15])
print(label_data[10])
print(word2id(label_data[10], vocab))


# ### 2.3 音频数据处理
# 
# 音频数据处理，只需要获得对应的音频文件名，然后提取所需时频图即可。
# 
# 其中`compute_fbank`时频转化的函数在前面已经定义好了。

# In[17]:
fbank = compute_fbank(wav_lst[0])
print(fbank.shape)


# In[18]:
# plt.imshow(fbank.T, origin = 'lower')
# plt.show()


# #### 由于声学模型网络结构原因（3个maxpooling层），我们的音频数据的每个维度需要能够被8整除。

# In[ ]:
fbank = fbank[:fbank.shape[0]//8*8, :]


# In[20]:
print(fbank.shape)


# #### 总结：
# - 对音频数据进行时频转换
# - 转换后的数据需要各个维度能够被8整除
# 
# ### 2.4 数据生成器
# #### 确定batch_size和batch_num

# In[ ]:
total_nums = 10000
batch_size = 4
batch_num = total_nums // batch_size


# #### shuffle
# 打乱数据的顺序，我们通过查询乱序的索引值，来确定训练数据的顺序

# In[ ]:


from random import shuffle
shuffle_list = [i for i in range(10000)]
shuffle(shuffle_list)


# #### generator
# batch_size的信号时频图和标签数据ID，存放到两个list中去,然后返回出来

# In[ ]:


def get_batch(batch_size, shuffle_list, wav_lst, label_data, vocab):
    for i in range(10000//batch_size):
        wav_data_lst = []
        label_data_lst = []
        begin = i * batch_size
        end = begin + batch_size
        sub_list = shuffle_list[begin:end]
        for index in sub_list:
            fbank = compute_fbank(wav_lst[index])
            fbank = fbank[:fbank.shape[0] // 8 * 8, :]
            label = word2id(label_data[index], vocab)
            wav_data_lst.append(fbank)
            label_data_lst.append(label)
        yield wav_data_lst, label_data_lst

batch = get_batch(4, shuffle_list, wav_lst, label_data, vocab)


# In[24]:


wav_data_lst, label_data_lst = next(batch)
for wav_data in wav_data_lst:
    print(wav_data.shape)
for label_data in label_data_lst:
    print(label_data)


# In[25]:
lens = [len(wav) for wav in wav_data_lst]
print(max(lens))
print(lens)


# #### padding
# 然而，每一个batch_size内的数据有一个要求，就是需要构成成一个tensorflow块，这就要求每个样本数据形式是一样的。
# 除此之外，ctc需要获得的信息还有输入序列的长度。
# 这里输入序列经过卷积网络后，长度缩短了8倍，因此我们训练实际输入的数据为wav_len//8。
# - padding wav data
# - wav len // 8 （网络结构导致的）

# In[26]:
def wav_padding(wav_data_lst):
    wav_lens = [len(data) for data in wav_data_lst]
    wav_max_len = max(wav_lens)
    wav_lens = np.array([leng//8 for leng in wav_lens])
    new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 200, 1))
    for i in range(len(wav_data_lst)):
        new_wav_data_lst[i, :wav_data_lst[i].shape[0], :, 0] = wav_data_lst[i]
    return new_wav_data_lst, wav_lens

pad_wav_data_lst, wav_lens = wav_padding(wav_data_lst)
print(pad_wav_data_lst.shape)# (4, 1104, 200, 1) 代表： bach_size=4  max_len=1104每个batch会变动的,bach里面最长的  ,特征向量=200，
print(wav_lens)


# 同样也要对label进行padding和长度获取，不同的是数据维度不同，且label的长度就是输入给ctc的长度，不需要额外处理
# - label padding
# - label len

# In[27]:
def label_padding(label_data_lst):
    label_lens = np.array([len(label) for label in label_data_lst])
    max_label_len = max(label_lens)
    new_label_data_lst = np.zeros((len(label_data_lst), max_label_len))
    for i in range(len(label_data_lst)):
        new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
    return new_label_data_lst, label_lens

pad_label_data_lst, label_lens = label_padding(label_data_lst)
print(pad_label_data_lst.shape)
print(label_lens)


# - 用于训练格式的数据生成器

# In[ ]:
def data_generator(batch_size, shuffle_list, wav_lst, label_data, vocab):
    for i in range(len(wav_lst)//batch_size):
        wav_data_lst = []
        label_data_lst = []
        begin = i * batch_size
        end = begin + batch_size
        sub_list = shuffle_list[begin:end]
        for index in sub_list:
            fbank = compute_fbank(wav_lst[index])
            pad_fbank = np.zeros((fbank.shape[0]//8*8+8, fbank.shape[1]))
            pad_fbank[:fbank.shape[0], :] = fbank
            label = word2id(label_data[index], vocab)
            wav_data_lst.append(pad_fbank)
            label_data_lst.append(label)
        pad_wav_data, input_length = wav_padding(wav_data_lst)
        pad_label_data, label_length = label_padding(label_data_lst)
        inputs = {'the_inputs': pad_wav_data,
                  'the_labels': pad_label_data,
                  'input_length': input_length,
                  'label_length': label_length,
                 }
        outputs = {'ctc': np.zeros(pad_wav_data.shape[0],)} #pad_wav_data.shape[0] = ？bach_size
        yield inputs, outputs


# ## 3. 模型搭建
# 
# 训练输入为时频图，标签为对应的拼音标签，如下所示：
# 
# 
# 搭建语音识别模型，采用了 CNN+CTC 的结构。
# ![dfcnn.jpg](attachment:dfcnn.jpg)

# In[29]:
import keras
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D
from keras.layers import Reshape, Dense, Lambda
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from keras.utils import multi_gpu_model


# - 定义3*3的卷积层

# In[ ]:


def conv2d(size):
    return Conv2D(size, (3,3), use_bias=True, activation='relu',
        padding='same', kernel_initializer='he_normal')


# - 定义batch norm层

# In[ ]:


def norm(x):
    return BatchNormalization(axis=-1)(x)


# - 定义最大池化层，数据的后两维维度都减半

# In[ ]:


def maxpool(x):
    return MaxPooling2D(pool_size=(2,2), strides=None, padding="valid")(x)


# - dense层

# In[ ]:


def dense(units, activation="relu"):
    return Dense(units, activation=activation, use_bias=True,
        kernel_initializer='he_normal')


# - 由cnn + cnn + maxpool构成的组合

# In[ ]:


# x.shape=(none, none, none)
# output.shape = (1/2, 1/2, 1/2)
def cnn_cell(size, x, pool=True):
    x = norm(conv2d(size)(x))
    x = norm(conv2d(size)(x))
    if pool:
        x = maxpool(x)
    return x


# - **添加CTC损失函数，由backend引入**
# 
# **注意：CTC_batch_cost输入为：**
# 
# - **labels** 标签：[batch_size, l]
# - **y_pred** cnn网络的输出：[batch_size, t, vocab_size]
# - **input_length** 网络输出的长度：[batch_size]
# - **label_length** 标签的长度：[batch_size]

# In[ ]:


def ctc_lambda(args):
    labels, y_pred, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# ### **搭建cnn+dnn+ctc的声学模型**

# In[ ]:


class Amodel():
    """docstring for Amodel."""
    def __init__(self, vocab_size):
        super(Amodel, self).__init__()
        self.vocab_size = vocab_size
        self._model_init()
        self._ctc_init()
        self.opt_init()

    def _model_init(self):
        self.inputs = Input(name='the_inputs', shape=(None, 200, 1))
        self.h1 = cnn_cell(32, self.inputs)
        self.h2 = cnn_cell(64, self.h1)
        self.h3 = cnn_cell(128, self.h2)
        self.h4 = cnn_cell(128, self.h3, pool=False)
        # 200 / 8 * 128 = 3200
        self.h6 = Reshape((-1, 3200))(self.h4)
        self.h7 = dense(256)(self.h6)
        self.outputs = dense(self.vocab_size, activation='softmax')(self.h7)
        self.model = Model(inputs=self.inputs, outputs=self.outputs)

    def _ctc_init(self):
        self.labels = Input(name='the_labels', shape=[None], dtype='float32')
        self.input_length = Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = Input(name='label_length', shape=[1], dtype='int64')
        self.loss_out = Lambda(ctc_lambda, output_shape=(1,), name='ctc')            ([self.labels, self.outputs, self.input_length, self.label_length])
        self.ctc_model = Model(inputs=[self.labels, self.inputs,
            self.input_length, self.label_length], outputs=self.loss_out)

    def opt_init(self):
        opt = Adam(lr = 0.0008, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01, epsilon = 10e-8)
        #self.ctc_model=multi_gpu_model(self.ctc_model,gpus=2)
        self.ctc_model.compile(loss={'ctc': lambda y_true, output: output}, optimizer=opt)


# In[37]:


am = Amodel(1176)#词汇数量？是的，在后面打印出来了，这里只是测试下模型代码，后面真正用的时候不是写死的。
# am.ctc_model.summary()


# ## 4. 开始训练
# 
# 这样训练所需的数据，就准备完毕了，接下来可以进行训练了。我们采用如下参数训练：
# - batch_size = 4
# - batch_num = 10000 // 4
# - epochs = 1

# - **准备训练数据，shuffle是为了打乱训练数据顺序**

# In[ ]:
total_nums = 10000
# total_nums = 100
batch_size = 4
batch_num = total_nums // batch_size
# epochs = 50
epochs = 5


# In[39]:


source_file = './dataset/data_thchs30'
#source_file = 'data_thchs30'
label_lst=[]
wav_lst= []
label_lst, wav_lst = source_get(source_file)
label_data = gen_label_data(label_lst[:total_nums])
vocab = mk_vocab(label_data)
vocab_size = len(vocab)

print(vocab_size)

shuffle_list = [i for i in range(total_nums)]


# - 使用fit_generator

# - 开始训练

# In[40]:


am = Amodel(vocab_size)

for k in range(epochs):
    print('this is the', k+1, 'th epochs trainning !!!')
    #shuffle(shuffle_list)
    batch = data_generator(batch_size, shuffle_list, wav_lst, label_data, vocab)
    am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=1)


# In[ ]:
def decode_ctc(num_result, num2word):
	result = num_result[:, :, :]
	in_len = np.zeros((1), dtype = np.int32)
	in_len[0] = result.shape[1];
	r = K.ctc_decode(result, in_len, greedy = True, beam_width=10, top_paths=1)
	r1 = K.get_value(r[0][0])
	r1 = r1[0]
	text = []
	for i in r1:
		text.append(num2word[i])
	return r1, text


# In[42]:
# 测试模型 predict(x, batch_size=None, verbose=0, steps=None)
batch = data_generator(1, shuffle_list, wav_lst, label_data, vocab)
for i in range(10):
  # 载入训练好的模型，并进行识别
  inputs, outputs = next(batch)
  x = inputs['the_inputs']
  y = inputs['the_labels'][0]
  result = am.model.predict(x, steps=1)
  # 将数字结果转化为文本结果
  result, text = decode_ctc(result, vocab)
  print('数字结果： ', result)
  print('文本结果：', text)
  print('原文结果：', [vocab[int(i)] for i in y])

