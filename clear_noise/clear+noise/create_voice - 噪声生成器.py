import os
import soundfile
import math
import random
import librosa
import fnmatch
import numpy as np
import soundfile as sf
import shutil

def find_files(directory, pattern='*.wav'):
    """返回当前目录(包括子目录)的所有wav文件
    :return:    返回wav文件的目录列表
    """
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):     # 实现列表特殊字符的过滤或筛选,返回符合匹配“.wav”字符列表
            files.append(os.path.join(root, filename))
    return files


def find_files_and_add_papa_title(directory, pattern='*.wav'):
    """返回当前目录(包括子目录)的所有wav文件,并按照ASRT的要求，生成两个列表文件，一个包括文件的完整路径，一个包括文件的拼音标签
    按照DEEP的要求，生成两个
    :return:    返回wav文件的目录列表
    """
    file_wav_list=[]
    file_symal_lable=[]
    deep_file_pinyin_lujin_lable=[]
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):     # 实现列表特殊字符的过滤或筛选,返回符合匹配“.wav”字符列表
            with open(os.path.join(root, filename+".txt"), "w+",encoding="utf-8") as fp:
                fp.write("pa3 pa3 pa3 pa3 pa3 pa3 pa3 pa3 pa3 pa3")
            files.append(os.path.join(root, filename))

            relpath =  os.path.join(root, filename)
            relpath = os.sep.join(relpath.split(os.sep)[3:])
            file_wav_list.append(filename+"\t"+relpath+"\n")
            file_symal_lable.append(filename+"\tpa3 pa3 pa3 pa3 pa3 pa3 pa3 pa3 pa3 pa3\t潖 潖 潖 潖 潖 潖 潖 潖 潖 潖\n")
            deep_file_pinyin_lujin_lable.append(relpath+"\tpa3 pa3 pa3 pa3 pa3 pa3 pa3 pa3 pa3 pa3\t潖 潖 潖 潖 潖 潖 潖 潖 潖 潖\n")
            #
            # file_wav_list.append(filename+" "+os.path.join(root, filename)+"\n")
            # file_symal_lable.append(filename+"\tpa3 pa3 pa3 pa3 pa3 pa3 pa3 pa3 pa3 pa3\t潖 潖 潖 潖 潖 潖 潖 潖 潖 潖\n")
            # deep_file_pinyin_lujin_lable.append(os.path.join(root, filename)+"\tpa3 pa3 pa3 pa3 pa3 pa3 pa3 pa3 pa3 pa3\t潖 潖 潖 潖 潖 潖 潖 潖 潖 潖\n")

    with open("mydata_train.wav" + ".lst", "w+", encoding="utf-8") as fp:
        fp.write(''.join(file_wav_list))#列表转化为字符串，用join()方法
    with open("mydata_train.sylable" + ".txt", "w+", encoding="utf-8") as fp:
        fp.write(''.join(file_symal_lable))
    with open("mydata_train" + ".txt", "w+", encoding="utf-8") as fp: #deep用的文件
        fp.write(''.join(deep_file_pinyin_lujin_lable))

    shutil.copy("mydata_train.txt"  , "../../data/mydata_train.txt")
    return files

#做完后，要手动把列表文件的前面3个路径用列模式删除。
noise_wav_file = find_files_and_add_papa_title("../../dataset/my_hunyin_papa_dataset", pattern='*.wav')
# 方法一
# def add_noise(clear_wav, noise_wav, new_wav_name):
#     clear_wav, fs = librosa.load(clear_wav, sr=16000)
#     noise_wav, Fs = librosa.load(noise_wav, sr=16000)
#     list = []
#     numpy.array(noise_wav, dtype=float32)
#     list.append(noise_wav)
#     list.re
#     while  len(noise_wav) < len(clear_wav):
#         # list.append(noise_wav)
#         # noise_wav_three[0:len(noise_wav)-1] = noise_wav[0:len(noise_wav)-1]
#         # noise_wav_three[len(noise_wav):len(noise_wav)*2 -1 ] = noise_wav[0:len(noise_wav)-1]
#         # noise_wav = noise_wav_three
#         clear_wav[]
#
#
#
#     indx = np.random.randint(0, high=len(noise_wav) - len(clear_wav), size=None)
#     noise_wav = noise_wav[indx: indx + len(clear_wav)]
#
#
#
#     new_wav = clear_wav + noise_wav
#     # librosa.output.write_wav(new_wav_name, new_wav, 16000)
#     sf.write(new_wav_name, new_wav, 16000)
#     return


# 方法二#nosie_volum:噪音的缩放比例
def add_noise(clear_wav, noise_wav, new_wav_name , nosie_volum):
    clear_wav, fs = librosa.load(clear_wav, sr=16000)
    noise_wav, Fs = librosa.load(noise_wav, sr=16000)
    if len(noise_wav) < len(clear_wav):#noise短
        #clear_wav = np.random.choice(clear_wav, size=len(clear_wav), replace=True, p=None)
        clear_wav_list = np.split(clear_wav , [len(noise_wav)])
        clear_wav_list[0] = clear_wav_list[0]+ noise_wav*nosie_volum
        list0 = np.array(clear_wav_list[0])
        list1 = np.array(clear_wav_list[1])
        new_wav = np.append(list0 ,list1)


        # new_wav = clear_wav[0:len(noise_wav)] + noise_wav[0:len(noise_wav)]*0.1 #+ clear_wav[len(noise_wav):]
        # new_wav = np.concatenate(new_wav ,clear_wav[len(noise_wav):])
        # new_wav = clear_wav + noise_wav.flatten * 0.1
        # librosa.output.write_wav(new_wav_name, new_wav, 16000)
        sf.write(new_wav_name, new_wav, 16000)
    else :#noise长
        noise_wav = np.random.choice(noise_wav, size=len(clear_wav), replace=True, p=None)
        new_wav = clear_wav + noise_wav*nosie_volum
        # librosa.output.write_wav(new_wav_name, new_wav, 16000)
        sf.write(new_wav_name, new_wav, 16000)
    return

clear_wav_file = find_files("./clear", pattern='*.wav')

#pointsource这个数据集里面很多猫叫，拨号音，可能要手动去掉，系数0.1合适
noise_wav_file = find_files("../../dataset/Room Impulse Response and Noise Database/RIRS_NOISES/pointsource_noises", pattern='*.wav')

for i in clear_wav_file:
    print("纯净语音", i)
    for j in noise_wav_file:
        noise_filename = j
        print("噪音", noise_filename)
        clear_name = str(i.split("/")[-1]).rstrip(".WAV")
        noise_name = str(noise_filename.split("/")[-1])
        create_wav_name = "../../dataset/my_hunyin_papa_dataset/hunyin_pointsource_noises/" + clear_name + "_" + noise_name
        add_noise(i, noise_filename, create_wav_name , 0.1)

#这个部门数据有点像啪啪的声音 系数1合适
noise_wav_file = find_files("../../dataset/Room Impulse Response and Noise Database/RIRS_NOISES/real_rirs_isotropic_noises", pattern='*.wav')

for i in clear_wav_file:
    print("纯净语音", i)
    for j in noise_wav_file:
        noise_filename = j
        print("噪音", noise_filename)
        clear_name = str(i.split("/")[-1]).rstrip(".WAV")
        noise_name = str(noise_filename.split("/")[-1])
        create_wav_name = "../../dataset/my_hunyin_papa_dataset/hunyin_real_rirs_isotropic_noises/" + clear_name + "_" + noise_name
        add_noise(i, noise_filename, create_wav_name , 1)#1 #噪音的缩放比例

#系数2合适
noise_wav_file = find_files("../../dataset/Room Impulse Response and Noise Database/RIRS_NOISES/simulated_rirs", pattern='*.wav')
#文件太多，删除方法  ls | xargs -n 9 rm -rf
for i in clear_wav_file:
    print("纯净语音", i)
    for j in noise_wav_file:
        noise_filename = j
        print("噪音", noise_filename)
        clear_name = str(i.split("/")[-1]).rstrip(".WAV")
        noise_name = str(noise_filename.split("/")[-1])
        create_wav_name = "../../dataset/my_hunyin_papa_dataset/hunyin_simulated_rirs/" + clear_name + "_" + noise_name
        add_noise(i, noise_filename, create_wav_name ,2)#2 #噪音的缩放比例








