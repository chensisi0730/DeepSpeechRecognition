import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.io import wavfile
print(librosa.__version__)#librosa.output was removed in librosa version 0.8.0.
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号
fs = 16000

# def display_mfcc(audio):
#     '''
#
#     :param audio: 所需分析的音频文件路径
#     :return: 会打印出音频文件的mel频谱图
#     '''
#     y, _ = librosa.load(audio)
#     audio_time = librosa.get_duration(y)
#     # 音频时长
#     print(audio_time)
#     y1 = librosa.feature.melspectrogram(y=y, n_mels=128, fmax=8000)
#     mfcc = librosa.feature.mfcc(S=librosa.power_to_db(y1))
#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(mfcc, x_axis="time", y_axis="mel")#module 'librosa' has no attribute 'display'
#     plt.colorbar()
#     plt.title(audio)
#     plt.tight_layout()
#     plt.show()
# display_mfcc("./1A - 128 - Bounce Intro(djoffice).wav")

wav_data, _ = librosa.load("./1A - 128 - Bounce Intro(djoffice).wav", sr=fs, mono=True)
#sr-采样频率：sr=None表示使用音频原始采样，如果不给sr赋值则采用默认值22050
#duration-获取音频时长，已s为单位,会截断

# from scipy.io import wavfile
# wavfile.write("./love_illusion_20s_w.mp3", sr, y) # 写入音频的另一种方式


# y,sr = librosa.load("./1A - 128 - Bounce Intro(djoffice).wav", sr=fs, mono=True)
# # 通过改变采样率来改变音速，相当于播放速度X2
# sf.write("resample.wav",y,sr*2)


#加噪(noise)：为音频加入白噪声，呲呲响的那种哦！
#0.2听不清了  0.05能听到，但很吵  0.02很是吵  0.01:可以接受  0.002从波形上看，有一点噪音，听着也一点点
y, sr = librosa.load('test.wav')
wn = np.random.randn(len(y))
y = np.where(y != 0.0, y + 0.01 * wn, 0.0) # 噪声不要添加到0上！np.where(condition,x,y)condition为true取x,否者取y  org=0.02
wavfile.write("test_add_noise.wav", sr, y) # 写入音频


def add_noise1(x, w=0.004): ·1
    # w：噪声因子
    output = x + w * np.random.normal(loc=2, scale=4, size=len(x))#org:loc=0, scale=1
    return output
Augmentation = add_noise1(x=wav_data, w=0.004) #org:0.004


def add_noise2(x, snr):
    # snr：生成的语音信噪比
    P_signal = np.sum(abs(x) ** 2) / len(x)  # 信号功率
    P_noise = P_signal / 10 ** (snr / 10.0)  # 噪声功率
    return x + np.random.randn(len(x)) * np.sqrt(P_noise)
Augmentation2 = add_noise2(x=wav_data, snr=50)

# ########### 画图
plt.subplot(3, 2, 1)
plt.title("语谱图", fontsize=15)
plt.specgram(wav_data, Fs=16000, scale_by_freq=True, sides='default', cmap="jet")
plt.xlabel('秒/s', fontsize=15)
plt.ylabel('频率/Hz', fontsize=15)

plt.subplot(3, 2, 2)
plt.title("波形图", fontsize=15)
time = np.arange(0, len(wav_data)) * (1.0 / fs)
plt.plot(time, wav_data)
plt.xlabel('秒/s', fontsize=15)
plt.ylabel('振幅', fontsize=15)

# ########### 画图
plt.subplot(3, 2, 3)
plt.title("语谱图(加噪)", fontsize=15)
plt.specgram(Augmentation, Fs=16000, scale_by_freq=True, sides='default', cmap="jet")
plt.xlabel('秒/s', fontsize=15)
plt.ylabel('频率/Hz', fontsize=15)

plt.subplot(3, 2, 4)
plt.title("波形图(加噪)", fontsize=15)
time = np.arange(0, len(wav_data)) * (1.0 / fs)
plt.plot(time, Augmentation)
plt.xlabel('秒/s', fontsize=15)
plt.ylabel('振幅', fontsize=15)


# ########### 画图
plt.subplot(3, 2, 5)
plt.title("语谱图(加噪)", fontsize=15)
plt.specgram(Augmentation2, Fs=16000, scale_by_freq=True, sides='default', cmap="jet")
plt.xlabel('秒/s', fontsize=15)
plt.ylabel('频率/Hz', fontsize=15)

plt.subplot(3, 2, 6)
plt.title("波形图(加噪)", fontsize=15)
time = np.arange(0, len(wav_data)) * (1.0 / fs)
plt.plot(time, Augmentation2)
plt.xlabel('秒/s', fontsize=15)
plt.ylabel('振幅', fontsize=15)

plt.tight_layout()
plt.show()

