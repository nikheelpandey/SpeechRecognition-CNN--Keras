
# coding: utf-8

# # Speech Recognition Using CNN
# 
# In this notebook I have done two things:
# - Data Visualisation
# - Data Analysis
# - Deep Learning Model
# 
# Well I haven't made them all perfect as it was my first time to be working with audios.
# 
# I have focused my efforts on downsizing after padding/chopping the respective files.

# Lets start by importing the essential libraries that we have on the system

# In[13]:


import os
from os.path import isdir, join
from pathlib import Path
import pandas as pd

# Math
import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
import librosa

from sklearn.decomposition import PCA

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import librosa.display

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import pandas as pd

get_ipython().magic(u'matplotlib inline')


# So how does an audio (.wav) looks and how do we store a wave in an array. Well, we store the discreet values of the amplititude of the wave at the the different time interval. The freq(sample_rate) refers to the the number of times we the sample of amplititue in a single second. In this particular dataset the sample rate is 16000 Hz. 

# In[2]:


train_audio_path = "../speech_recognition_tf/train/audio/"
filename = '../speech_recognition_tf/train/audio/cat/012c8314_nohash_0.wav'
sample_rate,samples = wavfile.read(filename)


# Lets define a function which takes audio as input and reconstruct the sound wave

# In[3]:


# defining a function that calculates the spectrograms

def log_specgram(audio, sample_rate, ):
    window_size=20 
    step_size = 10
    esp = 1e-10
    nperseg = int(round(window_size*sample_rate/1e3))
    noverlap = int(round(step_size*sample_rate/1e3))
    
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + 1e-10)


freqs, times, spectrogram = log_specgram(samples, sample_rate)


# Now we are going to visualise the sample audio .wav file that we read earlier 

# In[4]:


# freqs, times, spectrogram = specgram(samples, sample_rate)
fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave of ' + filename)
ax1.set_ylabel('Amplitude')
ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)


ax2= fig.add_subplot(212)
ax2.imshow(spectrogram.T, aspect= "auto", origin="lower",
          extent= [times.min(), times.max(),
                  freqs.min(),freqs.max()])

ax2.set_yticks(freqs[::16])
ax2.set_xticks(times[::16])
ax2.set_title("Spectrogram of "+ str(filename))
ax2.set_ylabel("Freqs in Hz")
ax2.set_xlabel('seconds')


# Neat! You can acctually see the wave corresponding to the sound.
# 
# - Points that can be taken from this is that a large portion of the data is blank as can be seen in the raw wave as well as in spectogram. What a good scientist would do is to compress the relevent data to save training time and accuracy.
# 
# However I'm leaving this for future as I didn't had time while making this notebook. 

# In[5]:


dirs = [f for f in os.listdir(train_audio_path) if isdir(join(train_audio_path, f))]
dirs.sort()
print('Number of labels: ' + str(len(dirs)))


# so they have given 31 classes that the model needed to recognize.

# # Anomaly Detection
# 
# As for this case there are following cases of anomaly that's possible
# - Imbalance between the classes
# - Imbalance between the length of indivisual files
#   - Files can be greater than 1 sec
#   - Files can be less than 1 sec
# 
# 
# Lets check for inter-class balance

# In[14]:


# Calculate
number_of_recordings = []
for direct in dirs:
    waves = [f for f in os.listdir(join(train_audio_path, direct)) if f.endswith('.wav')]
    number_of_recordings.append(len(waves))

# Plot
data = [go.Histogram(x=dirs, y=number_of_recordings)]
trace = go.Bar(
    x=dirs,
    y=number_of_recordings,
    marker=dict(color = number_of_recordings, colorscale='Viridius', showscale=True
    ),
)
layout = go.Layout(
    title='Number of recordings in given label',
    xaxis = dict(title='Words'),
    yaxis = dict(title='Number of recordings')
)
py.iplot(go.Figure(data=[trace], layout=layout))


# In[8]:


x_ls = []
for i in dirs:
    annon = 0
    for j in os.listdir(str(train_audio_path)+i):
        
        if j.endswith(".wav"):
            sample_rate,samples = wavfile.read(str(train_audio_path)+(str(i)+"/"+j))
            if abs(samples.shape[0]) != abs(sample_rate):
                annon +=1
    per = float(annon)*100/len(os.listdir(str(train_audio_path)+str(i)))
    x_ls.append(per)
    print("Annon in {}: {}".format(i,str(per))+"%" )


# In[9]:


data = [go.Histogram(x=dirs, y=x_ls)]
trace = go.Bar(
    x=dirs,
    y=x_ls,
    marker=dict(color = x_ls , colorscale='Viridius', showscale=True
    ),
)
layout = go.Layout(
    title='Number of recordings in given label',
    xaxis = dict(title='Words'),
    yaxis = dict(title='percentage of anomaly')
)
py.iplot(go.Figure(data=[trace], layout=layout))


# In[10]:


def custom_fft(y, fs):
    T = 1.0 / fs
    N = y.shape[0]
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    # FFT is simmetrical, so we take just the first half
    # FFT is also complex, to we take just the real part (abs)
    vals = 2.0/N * np.abs(yf[0:N//2])
    return xf, vals


# In[ ]:


L=16000
def pad_audio(samples):
    L = 16000
    if len(samples) >= L: return samples
    else: return np.pad(samples, pad_width=(L - len(samples), 0), mode='constant', constant_values=(0, 0))

def chop_audio(samples, L=16000, num=20):
    for i in range(num):
        beg = np.random.randint(0, len(samples) - L)
        return (samples[beg: beg + L])


# In[ ]:


downgarde_sample_rate = 8000
x_train, y_train = [], []

for i in dirs:
    for j in os.listdir(str(train_audio_path)+i):
        if j.endswith(".wav"):
            sample_rate,samples = wavfile.read(str(train_audio_path)+(str(i)+"/"+j))
            samples = pad_audio(samples)
            
            
            if len(samples)> 16000:
                samples = chop_audio(samples)
#             print(samples.shape[0])

            resampled = signal.resample(samples,  8000)
            _, _, specgram = log_specgram(resampled, 
                                  sample_rate=downgarde_sample_rate)
            
            x_train.append(specgram)
            y_train.append(i)
            
                


# In[ ]:


len(x_train) == len(y_train)


# In[ ]:


x_train = np.array(x_train)
x_train = x_train.reshape(tuple(list(x_train.shape) + [1]))


# In[ ]:


legal_labels = 'yes no up down left right on off stop go silence unknown'.split()
def label_transform(labels):
    nlabels = []
    for label in labels:
        if label == '_background_noise_':
            nlabels.append('silence')
        elif label not in legal_labels:
            nlabels.append('unknown')
        else:
            nlabels.append(label)
    return pd.get_dummies(pd.Series(nlabels))


# In[ ]:


y_train = label_transform(y_train)


# In[ ]:


label_index = y_train.columns.values
y_train = y_train.values
y_train = np.array(y_train)


# In[ ]:


from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
import keras


# In[ ]:


input_shape = (99, 81, 1)
nclass = 12
inp = Input(shape=input_shape)
norm_inp = BatchNormalization()(inp)
img_1 = Convolution2D(8, kernel_size=2, activation=activations.relu)(norm_inp)
img_1 = Convolution2D(8, kernel_size=2, activation=activations.relu)(img_1)
img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
img_1 = Dropout(rate=0.2)(img_1)
img_1 = Convolution2D(16, kernel_size=3, activation=activations.relu)(img_1)
img_1 = Convolution2D(16, kernel_size=3, activation=activations.relu)(img_1)
img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
img_1 = Dropout(rate=0.2)(img_1)
img_1 = Convolution2D(32, kernel_size=3, activation=activations.relu)(img_1)
img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
img_1 = Dropout(rate=0.2)(img_1)
img_1 = Flatten()(img_1)

dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(img_1))
dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(dense_1))
dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)

model = models.Model(inputs=inp, outputs=dense_1)
opt = optimizers.Adam()

model.compile(optimizer=opt, loss=losses.binary_crossentropy)
model.summary()

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, random_state=2017)
model.fit(x_train, y_train, batch_size=16, validation_data=(x_valid, y_valid), epochs=3, shuffle=True, verbose=2)




# In[ ]:


model.save(os.path.join(train_audio_path, 'cnn.model'))


# In[ ]:




