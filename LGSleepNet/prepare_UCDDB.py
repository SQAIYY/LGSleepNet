import os
import numpy as np
import mxnet as mx
import argparse
import glob
import math
import ntpath

import shutil
import urllib
# import urllib2

from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from mne.io import concatenate_raws, read_raw_edf
import dhedfreader
import xml.etree.ElementTree as ET

###############################
EPOCH_SEC_SIZE = 30

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="../data/UCDDB/",
                        help="File path to the PSG and annotation files.")
    parser.add_argument("--output_dir", type=str, default="../data/output_data/UCDDB/",
                        help="Directory where to save numpy files outputs.")
    parser.add_argument("--select_ch", type=str, default="C3A2",
                        help="The selected channel")
    args = parser.parse_args()

    # Output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    # Select channel
    select_ch = args.select_ch

    # Read raw and annotation EDF files
    psg_fnames = glob.glob(os.path.join(args.data_dir, "*.edf"))
    ann_fnames = glob.glob(os.path.join(args.data_dir, "*stage.txt"))
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)
    #print(psg_fnames)
    #print(ann_fnames)
    for file_id in range(len(psg_fnames)):
        print(psg_fnames[file_id])
        #read_record = mx.recordio.MXRecordIO(psg_fnames[i], "r")
        #print(read_record.read())
        raw = read_raw_edf(psg_fnames[file_id], preload=True, stim_channel=None)
        #sampling_rate = raw.info['sfreq']
        sampling_rate = 100.0
        raw = raw.resample(100, npad="auto")
        #print(raw.info)

        #print(sampling_rate)
        raw_ch_df = raw.to_data_frame(scalings=sampling_rate)[select_ch]
        raw_ch_df = raw_ch_df.to_frame()
        #print(len(raw_ch_df))
        raw_ch_df.set_index(np.arange(len(raw_ch_df)))
        ###################################################
        labels = []
        # Read annotation and its header

        with open(ann_fnames[file_id], "r") as f:  # 打开文件
            data = f.read()  # 读取文件
        #print(data)
        # 遍历，将数据全部放入列表中
        r = []
        for x in data:
            r.append(x.strip())
        r = list(filter(None, r))
        r = [int(x) for x in r]


        faulty_File = 0
        for i in range(len(r)):
            lbl = r[i]
            if lbl == 4:  # make stages N3, N4 same as N3
                labels.append(3)
            elif lbl == 5:
                labels.append(3)
            elif lbl == 2:
                labels.append(1)
            elif lbl == 3:
                labels.append(2)
            elif lbl == 1:  # Assign label 4 for REM stage
                labels.append(4)
            elif lbl == 0:
                labels.append(0)
            elif lbl > 5:
                labels.append(-1)


        labels = np.asarray(labels)
        # Remove movement and unknown stages if any
        raw_ch = raw_ch_df.values
        #print(raw_ch.shape)
        #print(labels.shape)

        # Verify that we can split into 30-s epochs
        if len(raw_ch) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
            L = int((len(raw_ch)/(EPOCH_SEC_SIZE * sampling_rate))) * int(EPOCH_SEC_SIZE * sampling_rate)
            raw_ch = raw_ch[0:L]
            #print(raw_ch.shape)
            #raise Exception("Something wrong")
        n_epochs = len(raw_ch) / (EPOCH_SEC_SIZE * sampling_rate)

        # Get epochs and their corresponding labels
        x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
        y = labels.astype(np.int32)
        index = np.where(y == -1)
        y = np.delete(y, index)
        x = np.delete(x, index, axis=0)

        print(x.shape)
        print(y.shape)
        # Select on sleep periods
        w_edge_mins = 30
        nw_idx = np.where(y != 0)[0]
        start_idx = nw_idx[0] - (w_edge_mins * 2)
        end_idx = nw_idx[-1] + (w_edge_mins * 2)
        if start_idx < 0: start_idx = 0
        if end_idx >= len(y): end_idx = len(y) - 1
        select_idx = np.arange(start_idx, end_idx + 1)
        print("Data before selection: {}, {}".format(x.shape, y.shape))
        x = x[select_idx]
        y = y[select_idx]
        print("Data after selection: {}, {}".format(x.shape, y.shape))

        # Saving as numpy files
        #print(os.path.basename(psg_fnames[file_id]))
        filename = os.path.basename(psg_fnames[file_id]).replace(".edf", ".npz")
        save_dict = {
            "x": x,
            "y": y,
            "fs": sampling_rate
        }
        np.savez(os.path.join(args.output_dir, filename), **save_dict)
        print(" ---------- Done this file ---------")


if __name__ == "__main__":
    main()