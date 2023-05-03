# AICITY 2023 Challenge ([Track5](https://www.aicitychallenge.org/)) -- Team : NYCU-Road Beast

## System workflow

<div align="center">
    <a href="./">
        <img src="./figure/system workflow.png" width="80%"/>
    </a>
</div>

## Training

Data preparation

```shell
python prepare_dataset.py --data_path ./videos --label_path ./gt.txt --save_path ./train
```

Training model

``` shell
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 36 --data <data path (*.yaml)> --img 1280 1280 --cfg cfg/training/PRB_Series/PRB-FRN6-3PY.yaml --weights '' --name PRB-FRN6-3PY --hyp data/hyp.scratch.p5.yaml
```

## Pseudo label

Get the pseudo label by inference the trained model. The prediction will be stored in ./runs/test/<--name>/labels

```shell
python test.py --data <data path (*.yaml)> --img 1280 --batch 32 --conf 0.5 --iou 0.65 --save-txt --device 0 --weights prb-fpn.pt --name PRB-FRN6-3PY_Pseudo_label
```

Next step combine the prediction with ground truth label. Each frame of the file is stored in a different file.

```shell
python pseudo_label.py --pred_path <pred_label (*.txt)> --gt_path <pred_label (*.txt)> --save_path <save_path>
```
## Inference (only detection)
Trained weight can be downloaded ([here](https://drive.google.com/drive/folders/1NkIa2MUWFOcpTFnU3EeD34-XIhFjWGqS?usp=sharing)).

On video:
``` shell
python detect.py --weights <weight path (*.pt)> --conf 0.5 --img-size 1280 --source <video path (*.mp4)>
```

On image:
``` shell
python detect.py --weights <weight path (*.pt)> --conf 0.5 --img-size 1280 --source <image path (*.mp4)>
```

## Inference (tracking)
On video:
``` shell
 python mc_demo_prb.py --weights pretrained/AICity_best_New.pt --source <video path (*.mp4)> --save-txt --img-size 1280
```

You will get the submmision file in 'runs/detect/exp*/labels/AI_result.txt'

## Reference 
This code is based on [PRBNet_Pytorch](https://github.com/pingyang1117/PRBNet_PyTorch)
