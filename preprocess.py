import os
import sys
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from itertools import repeat
from librosa.core import load, stft
from librosa.feature import mfcc
from librosa.util import frame
import pyworld as world
import pysptk as sptk
import numpy as np
from utils import repeat_last_padding, encoder, np_mulaw_quantized
from sklearn.preprocessing import StandardScaler
import argparse
import audio
def get_features(filename, *, winlen, winstep, n_mcep, mcep_alpha, minf0, maxf0, type):
    wav, sr = load(filename, sr=None)

    # get f0
    x = wav.astype(float)
    _f0, t = world.harvest(x, sr, f0_floor=minf0, f0_ceil=maxf0, frame_period=winstep * 1000)
    f0 = world.stonemask(x, _f0, t, sr)

    window_size = int(sr * winlen)
    hop_size = int(sr * winstep)

    # get mel
    if type == 'mcc':
        spec = world.cheaptrick(x, f0, t, sr, f0_floor=minf0)
        h = sptk.sp2mc(spec, n_mcep - 1, mcep_alpha).T
    else:
        h = mfcc(x, sr, n_mfcc=n_mcep, n_fft=window_size, hop_length=hop_size)
    h = np.vstack((h, f0))
    maxlen = len(x) // hop_size + 2
    h = repeat_last_padding(h, maxlen)
    id = os.path.basename(filename).replace(".wav", "")
    return (id, x, h)


def calc_stats(npzfile, out_dir):
    scaler = StandardScaler()
    data_dict = np.load(npzfile)
    for name, x in data_dict.items():
        if name[-2:] == '_h':
            scaler.partial_fit(x.T)

    mean = scaler.mean_
    scale = scaler.scale_

    np.savez(os.path.join(out_dir, 'scaler.npz'), mean=np.float32(mean), scale=np.float32(scale))


def preprocess_cmu(wav_dir, output, *, q_channels, winlen, winstep, n_mcep, mcep_alpha, minf0, maxf0, type):
    in_dir = os.path.join(wav_dir)
    out_dir = os.path.join(output)
    train_data = os.path.join(out_dir, 'train.npz')
    test_data = os.path.join(out_dir, 'test.npz')
    os.makedirs(out_dir, exist_ok=True)

    files = [os.path.join(in_dir, f) for f in os.listdir(in_dir)]
    files.sort()
    train_files = files[:1032]
    test_files = files[1032:]

    feature_fn = partial(get_features, winlen=winlen, winstep=winstep, n_mcep=n_mcep, mcep_alpha=mcep_alpha,
                         minf0=minf0, maxf0=maxf0, type=type)
    n_workers = cpu_count() // 2
    print("Running", n_workers, "processes.")

    data_dict = {}
    enc = encoder(q_channels)
    print("Processing training data ...")
    with ProcessPoolExecutor(n_workers) as executor:
        futures = [executor.submit(feature_fn, f) for f in train_files]
        for future in tqdm(futures):
            name, data, feature = future.result()
            data_dict[name] = enc(data).astype(np.uint8)
            data_dict[name + '_h'] = feature
    np.savez(train_data, **data_dict)

    data_dict = {}
    print("Processing test data ...")
    with ProcessPoolExecutor(n_workers) as executor:
        futures = [executor.submit(feature_fn, f) for f in test_files]
        for future in tqdm(futures):
            name, data, feature = future.result()
            data_dict[name] = enc(data).astype(np.uint8)
            data_dict[name + '_h'] = feature
    np.savez(test_data, **data_dict)

    calc_stats(train_data, out_dir)


def _process_wav(file_list, wav_dir, outfile, winlen, winstep, n_mcep, mcep_alpha, minf0, maxf0, q_channels, vocoderinput):
    data_dict = {}
    enc = encoder(q_channels)
    for f in tqdm(file_list):
        file = os.path.join(wav_dir, f)
        wav, sr = load(file, sr=None)

        x = wav.astype(float)
        _f0, t = world.harvest(x, sr, f0_floor=minf0, f0_ceil=maxf0,
                               frame_period=winstep * 1000)  # can't adjust window size
        f0 = world.stonemask(x, _f0, t, sr)

        window_size = int(sr * winlen)
        hop_size = int(sr * winstep)
        # get mel
        if vocoderinput == 'mcc':
            nfft = 2 ** (window_size - 1).bit_length()
            spec = np.abs(stft(x, n_fft=nfft, hop_length=hop_size, win_length=window_size, window='blackman')) ** 2
            h = sptk.mcep(spec, n_mcep - 1, mcep_alpha, eps=-60, etype=2, itype=4).T
        else:
            h = mfcc(x, sr, n_mfcc=n_mcep, n_fft=int(sr * winlen), hop_length=int(sr * winstep))
        h = np.vstack((h, f0))
        # mulaw encode
        wav = enc(x).astype(np.uint8)

        id = os.path.basename(f).replace(".wav", "")
        data_dict[id] = wav
        data_dict[id + "_h"] = h
    np.savez(outfile, **data_dict)


def _process_wav_melspectrogram(file_list, wav_dir, out_dir, q_channels=256):
    data_dict = {}

    for file_id in tqdm(file_list):
        filepath = os.path.join(wav_dir, file_id+'.wav')
        wav = audio.load_wav(filepath)
        if len(wav) < 5000:
            wav = np.tile(wav, ceil(5000/len(wav)))
        melspectrogram = audio.melspectrogram(wav)

        wav_quantized = np_mulaw_quantized(wav, q_channels) 

        data_dict[file_id] = wav_quantized
        data_dict[file_id+'_h'] = melspectrogram
    np.savez(out_dir, **data_dict)

def get_wavfiles_list(listfile):
    with open(listfile, 'r') as f:
        all_lines = f.readlines()
    wav_files = [line.split('|')[0] for line in all_lines]
    return wav_files

def melspectrogram_preprocess(in_dir, out_dir, vocoderinput="melspectrogram", **kwargs):
    os.makedirs(out_dir, exist_ok=True)

    wav_dir = os.path.join(in_dir, "wavs")
    train_list_path = os.path.join(in_dir, "train.csv")
    test_list_path = os.path.join(in_dir, "test.csv")

    train_data = os.path.join(out_dir, 'train.npz')
    test_data = os.path.join(out_dir, 'test.npz')

    train_files = get_wavfiles_list(train_list_path)
    test_files = get_wavfiles_list(test_list_path)
    print("Processing testing data ...")
    _process_wav_melspectrogram(test_files, wav_dir, test_data, **kwargs)

    print("Processing training data ...")
    _process_wav_melspectrogram(train_files, wav_dir, train_data, **kwargs)

    calc_stats(train_data, out_dir)


if __name__ == '__main__':

    melspectrogram_preprocess(sys.argv[1],sys.argv[2], "melspectrogram", q_channels=256)
