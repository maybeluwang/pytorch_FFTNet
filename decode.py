import librosa
import torch
from torch.nn import functional as F
from tqdm import tqdm
import os
import argparse
import numpy as np
from torchaudio import save
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from datetime import datetime
from utils import decoder
from logmmse import logmmse
from preprocess import get_features
import audio
from hparams import hparams
parser = argparse.ArgumentParser()
parser.add_argument('--scaler_file', type=str, default=None)
parser.add_argument('--infile', type=str, default=None)
parser.add_argument('--save_path', type=str, default=None)
parser.add_argument('--data_dir', type=str, default='slt_mcc_data')
parser.add_argument('--feature_type', type=str, default='mcc')
parser.add_argument('--feature_dim', type=int, default=25, help='number of mcc coefficients')
parser.add_argument('--mcep_alpha', type=float, default=0.42, help='''all-pass filter constant.
                                                                   16khz: 0.42,
                                                                   10khz: 0.35,
                                                                   8khz: 0.31.''')
parser.add_argument('--window_length', type=float, default=0.025)
parser.add_argument('--window_step', type=float, default=0.01)
parser.add_argument('--minimum_f0', type=float, default=71)
parser.add_argument('--maximum_f0', type=float, default=800)
parser.add_argument('--q_channels', type=int, default=256, help='quantization channels')
parser.add_argument('--interp_method', type=str, default='linear')
parser.add_argument('-c', type=float, default=2., help='a constant multiply before softmax.')
parser.add_argument('--model_file', type=str, default='slt_fftnet.pth')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--denoise', action='store_true')
parser.add_argument('--noise_std', type=float, default=0.005)


if __name__ == '__main__':
    args = parser.parse_args()
    net = torch.load(args.model_file)
    scaler = StandardScaler()
    scaler_info = np.load(args.scaler_file)
    scaler.mean_ = scaler_info['mean']
    scaler.scale_ = scaler_info['scale']
    filename = args.infile
    net.eval()
    if not args.cuda:
        net = net.cpu()
    else:
        net = net.cuda()

    print(args.model_file, "has", sum(p.numel() for p in net.parameters() if p.requires_grad), "of parameters.")

    with torch.no_grad():
        if args.infile is None:
            # haven't implement
            pass
        elif args.save_path is not None:
            x = audio.load_wav(filename)
            h = audio.melspectrogram(x)
            id = os.path.basename(filename).replace(".wav", "")
            h = scaler.transform(h.T).T
            # interpolation
            hopsize = hparams.frame_shift
            if args.interp_method == 'linear':
                xx = np.arange(h.shape[1]) * hopsize
                f = interp1d(xx, h, copy=False, axis=1, fill_value="extrapolate")
                h = f(np.arange(xx[-1]))
            elif args.interp_method == 'repeat':
                h = np.repeat(h, hopsize, axis=1)
            else:
                print("interpolation method", args.interp_method, "is not implemented.")
                exit(1)

            h = torch.from_numpy(h).unsqueeze(0).float()
            r_field = net.get_receptive_field()
            pred_dist = net.get_predict_distance()
            zcr = librosa.feature.zero_crossing_rate(x,frame_length = hparams.frame_length, hop_length = hparams.frame_shift)
            vad_curve = (zcr <=0.2)
            vad_curve = np.repeat(vad_curve, hopsize)

            output_buf = torch.empty(h.size(2)).long()
            h = F.pad(h, (r_field, 0))
            samples = torch.zeros(pred_dist).long()
            if args.cuda:
                h = h.cuda()
                samples = samples.cuda()

            net.init_buf()
            a = datetime.now().replace(microsecond=0)
            for pos in tqdm(range(r_field + pred_dist, h.size(2) + 1, pred_dist)):
                out_pos = pos - r_field - pred_dist
                decision = np.mean(vad_curve[out_pos:out_pos + pred_dist])
                if decision > 0.5:
                    samples = net.one_sample_generate(samples, h=h[:, :, :pos], c=args.c)
                else:
                    samples = net.one_sample_generate(samples, h=h[:, :, :pos])

                output_buf[out_pos:out_pos + pred_dist] = samples
            cost = datetime.now().replace(microsecond=0) - a
            dec = decoder(args.q_channels)
            generation = dec(output_buf)
            result = generation.cpu().numpy()
            if args.denoise:
                result = logmmse(result, hparams.sample_rate)
            
            audio.save_wav(result, args.save_path)
            print("Speed:", generation.size(0) / cost.total_seconds(), "samples/sec.")
            print('file saved in', args.save_path)
        else:
            print("Please enter output file name.")
