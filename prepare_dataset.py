import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import tqdm
import pandas as pd
import cv2
import sys
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./videos/*.mp4', help='videos path')
    parser.add_argument('--label_path', type=str, default='./gt.txt', help='labels path')
    parser.add_argument('--save_path', type=str, default='./train', help='save path')
    opt = parser.parse_args()

    video_paths = sorted(glob.glob(opt.data_path))
    labels = pd.read_csv(opt.label_path, delimiter = ",", header = None, names = ["video_id", "frame", "track_id", "bb_left", "bb_top", "bb_width", "bb_height", "class"])
    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)

    H = 1080
    W = 1920
    SAVE_IMG = False
    for video_idx in range(1, 2):#len(video_paths) + 1
            # save images
            video_path = video_paths[video_idx - 1]
            video = cv2.VideoCapture (video_path)
            success, image = video.read()
            count = 1
            while success:
                cv2.imwrite("{}/{}_{}.png".format(opt.save_path, video_idx, count), image)
                success, image = video.read()
                count += 1
            video.release()
            frame_id = []

            # save labels
            i_labels = labels.loc[labels["video_id"] == video_idx]
            for image_idx in range(i_labels.shape[0]):
                frame_i = i_labels.iloc[image_idx][1]
                if frame_i in frame_id:
                    continue
                else:
                    frame_id.append(frame_i)
                
                same_frame_label = i_labels.loc[i_labels["frame"] == frame_i]
                image_idx += same_frame_label.shape[0]
                
                with open("{}/{}_{}.txt".format(opt.save_path, video_idx, frame_i), "w") as f:
                    for i in range(same_frame_label.shape[0]):
                        bb_left = same_frame_label.iloc[i][3]
                        bb_top = same_frame_label.iloc[i][4]
                        bb_width = same_frame_label.iloc[i][5]
                        bb_height = same_frame_label.iloc[i][6]
                        class_i = same_frame_label.iloc[i][7]
                        if class_i in [1]:
                            print("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(0, (bb_left + bb_width / 2) / W, (bb_top + bb_height / 2) / H, bb_width / W, bb_height / H), file = f)
                        elif class_i in [2, 4, 6, 3, 5, 7]:
                            print("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(1, (bb_left + bb_width / 2) / W, (bb_top + bb_height / 2) / H, bb_width / W, bb_height / H), file = f)
