import os
import torchaudio
import torch
from scipy.fft import fft, fftfreq
import numpy as np
from scipy.signal import butter, sosfilt
import matplotlib.pyplot as plt
import shutil
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import TensorDataset


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    sos = butter(N=order, Wn = [lowcut, highcut], btype = 'bandpass', output = 'sos', fs = fs)
    y = sosfilt(sos, data)
    return y

def filter_and_split(filepath, resample_rate=4000):
    """A function used to split an audio file into filtered 2 second segments.
    
    The audio will be resampled to 4000 Hz and filtered with a second-order Butterworth
    bandpass filter with upper and lower cut-off frequencies of 400 and 25 Hz. Then the
    resulting array will be split into segments of 10000 items corresponding to 2 
    seconds of audio.

    Parameters
    ----------
    filepath: str, required
        The filepath for the audio file to split
    sample_rate: int, optional
        The rate to resample the waveform to

    Returns
    -------
    ndarray
        an array of arrays, each corresponding to a unique 2 second clip
    """

    # Convert wav file into waveform
    waveform, sample_rate = torchaudio.load(filepath)

    # Transform waveform into 4000 Hz sample rate
    transform = torchaudio.transforms.Resample(sample_rate, resample_rate)
    resampled_waveform = transform(waveform)

    # Filter the waveform
    filtered_waveform = butter_bandpass_filter(resampled_waveform, 25, 400, resample_rate, 2)

    # Cut final waveform into 2 second segments
    segment_len = resample_rate * 2  # segment length = resample_rate * seconds of audio = resample_rate * 2
    num_segments = filtered_waveform.size // segment_len
    return filtered_waveform.flatten()[:num_segments * segment_len].reshape(-1, segment_len)


def create_bispectrum(waveform, nperseg=256, sample_rate=4000):
    """Perform calculations to create a bispectrum image from the
    given waveform.

    Parameters
    ----------
    waveform: ndarray, required
        The waveform that the bispectrum image will be made from
    nperseg: int, optinal
        The final size of bispectrum, (nperseg, nperseg). Default is 256
    sample_rate: int, optional
        The sample rate of the given waveform, default is 4000 Hz.
    
    Returns
    -------
    tensor
        The bispectrum image of shape (1, nperseg, nperseg)
    """
    
    # Compute FFT
    X = fft(waveform)
    
    # Compute frequency array
    freqs = fftfreq(nperseg, 1/sample_rate)
    nfreqs = len(freqs)
    
    # Initialize bispectrum array
    bispec = np.zeros((nfreqs, nfreqs), dtype=complex)
    
    # Compute bispectrum for single segment
    for f1 in range(nfreqs):
        for f2 in range(nfreqs):
            f3 = f1 + f2
            if f3 < nfreqs:
                bispec[f1, f2] = X[f1] * X[f2] * np.conj(X[f3])
    
    # Normalize
    bispec = np.abs(bispec)
    bispec /= bispec.max()
    
    return torch.tensor(bispec.astype(np.float32), dtype=torch.float32).unsqueeze(0)


def sort_ICBHI():
    source_folder = "Audio Files\ICBHI Audio Files"
    names = {'pneumonia': 0, 'heart failure': 0, 'copd': 0, 'plueral effusion': 0, 'bron': 0, 'N': 0, 'asthma': 0, 'lung fibrosis': 0}
    for file_name in os.listdir(source_folder):
        source_path = os.path.join(source_folder, file_name)
        diag = file_name.split("_")[1].split(",")[0].lower()
        if diag == "n":
            diag = "N"
        if diag in names:
            destination_path = "Audio Files\\" + diag + "\ICBHI_" + str(names[diag]) + "_" + diag + ".wav"
            shutil.copy2(source_path, destination_path)
            names[diag] += 1

        
def generate_images(tl_directory):
    """This function takes a string to the top level directory of the training data
    files and produces two lists, one for the bispectrum images, and one for the 
    corresponding labels.

    Parameters
    ----------
    tl_directory: str, required
        A string representing the filepath to the top level directory of the training data
    
    Returns
    -------
    DataFrame
        A pandas dataframe with two columns, images and labels
    """

    x_data = []
    y_data = []
    for diagnosis in os.listdir(tl_directory):
        print("Working on directory: ", diagnosis)
        for audio in os.listdir(tl_directory + "/" + diagnosis):
            filepath = tl_directory + "/" + diagnosis + "/" + audio
            filtered_audio = filter_and_split(filepath)
            for segment in filtered_audio:
                bispec = create_bispectrum(segment)
                x_data.append(bispec)
                y_data.append(diagnosis)
    data = {"images": x_data, "labels": y_data}
    df = pd.DataFrame(data)
    return df
        

def prepare_dataframe(df):
    """Transfer the Pandas dataframe into a PyTorch DataSet
    
    Parameters
    ----------
    df: DataFrame, required
        The dataframe to convert. Should have two columns: images, labels.
        images should be a (1, 256, 256) Tensor
        labels should be a string

    Returns
    -------
    DataLoader
        A PyTorch dataloader of the data in df
    list[str]
        A list of labels where the index of each label corresponds to the int value of the one hot encoding
    """

    enc = OneHotEncoder()
    y = torch.tensor(enc.fit_transform(df[['labels']]).toarray(), dtype=torch.float32)
    encoding_mapping = enc.categories_[0].tolist()
    
    x = torch.stack(df['images'].to_list())

    ds = TensorDataset(x, y)

    return ds, encoding_mapping


def train_test_validate_split(df):
    test_df = df.groupby('labels').head(40)
    train_df = df.drop(test_df.index)

    validate_df = train_df.groupby('labels').head(20)
    train_df = train_df.drop(validate_df.index)

    train_df = train_df.groupby('labels').head(200).reset_index(drop=True)
    return train_df, test_df, validate_df

