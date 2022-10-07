import argparse
import os
import platform
import sys
import numpy as np
from pathlib import Path


import torch

from utils import general
from models.experimental import attempt_load
from utils.augmentations import letterbox

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def ParseOpts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opts = parser.parse_args()
    opts.imgsz *= 2 if len(opts.imgsz) == 1 else 1  # expand
    print_args(vars(opts))
    return opts


def SelectDevice(device: str):
    device = select_device(device)
    LOGGER.info(f'Select device: {device}')
    return device


def LoadModel(weights: str, device):
    model = attempt_load(weights, device=device, inplace=True, fuse=True)
    label_names = model.module.names if hasattr(model, 'module') else model.names
    stride = model.stride.numpy()
    LOGGER.info(label_names)
    return model, label_names, stride


def PrintSize(tag: str, x):
    def PrintImpl(x, s):
        if isinstance(x, torch.Tensor):
            s += str(x.size())
            return s
        elif isinstance(x, (list, tuple)):
            num = len(x)
            s += '['
            for i in range(num):
                s = PrintImpl(x[i], s)
                s += ', '
            s += ']'
        return s
    s = PrintImpl(x, '')
    print(tag, s)


def ProcImage(model, device, imgsz=640, conf_thres=0.3, iou_thres=0.5, classes=None, max_det=1000):
    # if webcam:  # WSL unsupported webcam
    #     cap = cv2.VideoCapture('rtsp://localhost:554/h264')
    #     assert cap.isOpened(), f'Failed to open camera {source}'
    path = 'data/images/zidane.jpg'
    im0 = cv2.imread(path)
    im = letterbox(im0, imgsz)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = im / 255.0  # 0 - 255 to 0.0 - 1.0
    im = np.ascontiguousarray(im)  # contiguous

    im = torch.from_numpy(im).to(device)
    im = im.float()
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    print('input tensor:', im.size())  # torch.Size([1, 3, 384, 640])

    pred = model(im)
    # [torch.Size([1, 15120, 6]), [torch.Size([1, 3, 48, 80, 6]), torch.Size([1, 3, 24, 40, 6]), torch.Size([1, 3, 12, 20, 6]), ], ]
    PrintSize('outputs:', pred)
    pred = pred[0]  # torch.Size([1, 15120, 6])

    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, max_det=max_det)  # list of tensor(n,6)
    PrintSize('nms results:', pred)  # [torch.Size([2, 6]), ]

    det = pred[0].cpu().numpy()  # [2, 6]
    print('bbox:\n', det)
    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
    print('rescaled bbox:\n', det)

    # draw bbox
    color = (0, 200, 110)
    lw = max(round(sum(im.shape) / 2 * 0.003), 2)  # line width
    for *xyxy, conf, cls in reversed(det):
        # bounding box
        p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        cv2.rectangle(im0, p1, p2, color, lw, lineType=cv2.LINE_AA)
        # label box
        label = f'{names[int(cls)]} {conf:.2f}'
        ft = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=ft)[0]
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(im0, p1, p2, color, -1, cv2.LINE_AA)  # filled
        # label text
        cv2.putText(im0,
                    label,
                    (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    lw / 3,
                    (255, 255, 255),
                    ft,
                    lineType=cv2.LINE_AA)
    cv2.imwrite('pred.png', im0)
    cv2.imshow(path, im0)
    cv2.waitKey(30 * 1000)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    LOGGER.setLevel('DEBUG')
    opts = ParseOpts()
    device = SelectDevice(opts.device)
    model, names, stride = LoadModel(opts.weights, opts.source, device)
    ProcImage(model, device)
