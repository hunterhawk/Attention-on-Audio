import librosa
import librosa.display
import os
import numpy as np
import matplotlib.pyplot as plt

train_root = 'D:\\学习资源\\DEEPSEA\\spectrogram\\AttentionPaper'
def to_log_S(fname, root):
    print(os.path.join(root, fname))
    y, sr = librosa.load(os.path.join(root, fname), mono=False)
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

display_spectogram(to_log_S('113-5-0-28.wav', train_root))