import os 
import wave
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
def to_log_S(fname):
    y, sr = librosa.load(fname)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    return log_S

def display_spectogram(log_S):
    sr = 22050
    plt.figure(figsize=(12,4))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    plt.title('mel power spectrogram')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    plt.show()

def specgram_gen(original_path, new_path):
    for dir_item in os.listdir(original_path):
        print(dir_item)
        next_dir_item = os.path.join(original_path, dir_item)
        print(next_dir_item)
        
        for sub_dir_item in os.listdir(next_dir_item):
            print(sub_dir_item)
            class_num = sub_dir_item.split("-")[1]
            # print(type(class_num), class_num)
            new_dir_item = os.path.join(new_path, class_num)
            if not os.path.exists(new_dir_item):
                os.mkdir(new_dir_item)
            file_name = os.path.join(next_dir_item, sub_dir_item)
            new_file_name = os.path.join(new_dir_item, sub_dir_item)
            print(file_name)
            print(new_file_name)

            # display_spectogram(to_log_S(file_name))
            wave_data, sample_rate = sf.read(file_name)
            wave_data = wave_data.transpose()
            # print(wave_data.shape[0])
            if wave_data.shape[0] == 2:
                channel_1 = wave_data[0, : ]
                channel_2 = wave_data[1, : ]

                channel_1 = channel_1 / (max(abs(channel_1)))
                channel_2 = channel_2 / (max(abs(channel_2)))
            else:
                channel_1 = wave_data
                channel_1 = channel_1 / (max(abs(channel_1)))
            if wave_data.shape[0] == 2:
                plt.specgram(channel_1, NFFT = 512, Fs = 44100, noverlap = 384)
                plt.axis('off')
                plt.axes().get_xaxis().set_visible(False)
                plt.axes().get_yaxis().set_visible(False)
            
                fig = plt.gcf()
                plt.savefig(new_file_name + '_1.jpg', bbox_inches = 'tight', pad_inches = 0)
                # plt.show()
                plt.clf()

                plt.specgram(channel_2, NFFT = 512, Fs = 44100, noverlap = 384)
                plt.axis('off')
                plt.axes().get_xaxis().set_visible(False)
                plt.axes().get_yaxis().set_visible(False)
            
                fig = plt.gcf()
                plt.savefig(new_file_name + '_2.jpg', bbox_inches = 'tight', pad_inches = 0)
                # plt.show()
                plt.clf()
            else:
                plt.specgram(channel_1, NFFT = 512, Fs = 44100, noverlap = 384)
                plt.axis('off')
                plt.axes().get_xaxis().set_visible(False)
                plt.axes().get_yaxis().set_visible(False)
            
                fig = plt.gcf()
                plt.savefig(new_file_name + '.jpg', bbox_inches = 'tight', pad_inches = 0)
                # plt.show()
                plt.clf()
            '''
            f = wave.open(file_name, "rb")
            params = f.getparams()
            # 声道/量化数/采样频率/采样点数
            nchannels, sampwidth, framerate, nframes = params[:4]
            # 读取音频，字符串格式
            strData = f.readframes(nframes)
            # 将字符串转化为int
            waveData = np.fromstring(strData, dtype = np.int16)
            # wave幅值归一化
            waveData = waveData * 1.0 / (max(abs(waveData)))

            plt.specgram(waveData, NFFT = 512, Fs=44100, noverlap=384)
            plt.axis('off')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)

            # fig = plt.gcf()
            plt.savefig(new_file_name + '.jpg', dpi = 1200, bbox_inches = 'tight', pad_inches = 0)
            plt.clf()
            '''
            
'''        
f = wave.open(file_name, "rb")
params = f.getparams()
# 声道/量化数/采样频率/采样点数
nchannels, sampwidth, framerate, nframes = params[:4]
strData = f.readframes(nframes)
waveData = np.fromstring(strData, dtype = np.int16)
waveData = waveData * 1.0 / (max(abs(waveData)))

plt.specgram(waveData, NFFT = 512, Fs=44100, noverlap=384)
plt.axis('off')
plt.axes().get_xaxis().set_visible(False)
plt.axes().get_yaxis().set_visible(False)

fig = plt.gcf()
plt.show()
'''
if __name__ == "__main__":
    original_path = "audio"
    new_path = "pygen_specgram"
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    specgram_gen(original_path, new_path)