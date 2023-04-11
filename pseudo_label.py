import glob
import os
import sys
import numpy as np
import tqdm
import argparse

def cal_overlap(bbox1, bbox2):
    bbox1_sx = bbox1[0] - bbox1[2] / 2
    bbox1_sy = bbox1[1] - bbox1[3] / 2
    bbox1_ex = bbox1[0] + bbox1[2] / 2
    bbox1_ey = bbox1[1] + bbox1[3] / 2

    bbox2_sx = bbox2[0] - bbox2[2] / 2
    bbox2_sy = bbox2[1] - bbox2[3] / 2
    bbox2_ex = bbox2[0] + bbox2[2] / 2
    bbox2_ey = bbox2[1] + bbox2[3] / 2

    inter_sx = min(bbox1_sx, bbox2_sx)
    inter_sy = min(bbox1_sy, bbox2_sy)
    inter_ex = max(bbox1_ex, bbox2_ex)
    inter_ey = max(bbox1_ey, bbox2_ey)

    inter_area = (inter_ex - inter_sx) * (inter_ey - inter_sy)
    bbox1_area = (bbox1_ex - bbox1_sx) * (bbox1_ey - bbox1_sy)
    bbox2_area = (bbox2_ex - bbox2_sx) * (bbox2_ey - bbox2_sy)

    return (bbox1_area + bbox2_area) / inter_area

def cal_iou(bbox1, bbox2):
    bbox1_sx = bbox1[0] - bbox1[2] / 2
    bbox1_sy = bbox1[1] - bbox1[3] / 2
    bbox1_ex = bbox1[0] + bbox1[2] / 2
    bbox1_ey = bbox1[1] + bbox1[3] / 2

    bbox2_sx = bbox2[0] - bbox2[2] / 2
    bbox2_sy = bbox2[1] - bbox2[3] / 2
    bbox2_ex = bbox2[0] + bbox2[2] / 2
    bbox2_ey = bbox2[1] + bbox2[3] / 2

    inter_area_sx = max(bbox1_sx, bbox2_sx)
    inter_area_sy = max(bbox1_sy, bbox2_sy)
    inter_area_ex = min(bbox1_ex, bbox2_ex)
    inter_area_ey = min(bbox1_ey, bbox2_ey)

    if inter_area_ex < inter_area_sx or inter_area_ey < inter_area_sy:
        return 0
    else:
        inter_area = (inter_area_ex - inter_area_sx) * (inter_area_ey - inter_area_sy)
        bbox1_area = (bbox1_ex - bbox1_sx) * (bbox1_ey - bbox1_sy)
        bbox2_area = (bbox2_ex - bbox2_sx) * (bbox2_ey - bbox2_sy)
        union_area = bbox1_area + bbox2_area - inter_area

        return inter_area / union_area

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str, default='.\\predict', help='prediction path')
    parser.add_argument('--gt_path', type=str, default='.\\train', help='ground truth path')
    parser.add_argument('--save_path', type=str, default='.\\pseudo', help='save path')
    opt = parser.parse_args()

    gt_label_paths = glob.glob(os.path.join(opt.gt_path, "*.txt"))
    pred_label_paths = glob.glob(os.path.join(opt.pred_path, "*.txt"))

    gt_label_paths.remove("{}\\classes.txt".format(opt.gt_path))
    pred_label_paths.remove("{}\\classes.txt".format(opt.pred_path))

    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)
        
    for gt_path in tqdm.tqdm([".\\Train_label\\Train_label\\100_107.txt"]):#gt_label_paths[:1]
        gt_label = np.loadtxt(gt_path)
        if len(gt_label.shape) == 1:
            gt_label = gt_label.reshape(-1, 5)
        case_name = gt_path.split("\\")[-1]
        if os.path.join(opt.pred_path, case_name) in pred_label_paths:
            # if gt and pred has output on same frame
            pred_label = np.loadtxt(os.path.join(opt.pred_path, case_name))
            if len(pred_label.shape) == 1:
                pred_label = pred_label.reshape(-1, 6)
            # remove the path from predict list
            pred_label_paths.remove(os.path.join(opt.pred_path, case_name))
            
            pseudo_label = gt_label

            pred_label_selected = pred_label
            gt_label_selected = gt_label
            duplicate = False
            for pred_i in range(pred_label_selected.shape[0]):
                for gt_i in range(gt_label_selected.shape[0]):
                    if cal_iou(pred_label_selected[pred_i, 1:], gt_label_selected[gt_i, 1:]) >= 0.5:
                        duplicate = True
                        break
                if not duplicate:
                    print("case_name : ", case_name, pred_label_selected[pred_i, :5])
                    pseudo_label = np.vstack([pseudo_label, pred_label_selected[pred_i, :5]])
                duplicate = False
            np.savetxt("{}\\{}".format(opt.save_path, case_name), pseudo_label, fmt = ["%1d", "%.6f", "%.6f", "%.6f", "%.6f"])
        else:
            # if only gt has label on the frame
            np.savetxt("{}\\{}".format(opt.save_path, case_name), gt_label, fmt = ["%1d", "%.6f", "%.6f", "%.6f", "%.6f"])
            pass

    # print(pred_label_paths)
    # scan the rest predict label
    for pred_path in tqdm.tqdm(pred_label_paths):
        pred_label = np.loadtxt(pred_path)
        case_name = pred_path.split("\\")[-1]
        np.savetxt("{}\\{}".format(opt.save_path, case_name), pred_label, fmt = ["%1d", "%.6f", "%.6f", "%.6f", "%.6f"])