import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

def filter(src):
    '''
    #img_in = './data/train/images/' + opt.input
    #src = cv2.imread(cv2.samples.findFile(img_in))
    #cv.imshow('Awal', src)
    if src is None:
        print('Could not open or find the image: ', opt.input)
        exit(0)
    '''
    lower_hue = 179
    lower_saturation = 255
    lower_value = 255
    lower_hue = 179
    lower_saturation = 255
    upper_value = 255
    hsv = cv2.cvtColor(src,cv2.COLOR_RGB2HSV)
    #cv.imshow('HSV', hsv)
    #tkinter.messagebox.showinfo(title=None, message=upper_hue, **options)
    mask = cv2.inRange(hsv, (lower_hue, lower_saturation, lower_value), (upper_hue, upper_saturation, upper_value))
    #cv.imshow('mask', mask)
    result = cv2.bitwise_and(src, src, mask=mask)
    cv2.imshow('Filter', result)
    return result
    #img_out = './data/train/images/FILTER_' + opt.input
    #cv2.imwrite(img_out, result)

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    img_filter = opt.img_filter
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    low_h = int(opt.low_hue)
    low_s = int(opt.low_saturation)
    low_v = int(opt.low_value)
    high_h = int(opt.high_hue)
    high_s = int(opt.high_saturation)
    high_v = int(opt.high_value)
    low_r = int(opt.low_red)
    low_g = int(opt.low_green)
    low_b = int(opt.low_blue)
    high_r = int(opt.high_red)
    high_g = int(opt.high_green)
    high_b = int(opt.high_blue)
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(img_size=imgsz, stride=stride, img_filter=img_filter, lower_red=low_r, lower_green=low_g, lower_blue=low_b, upper_red=high_r, upper_green=high_g, upper_blue=high_b, lower_hue=low_h, lower_saturation=low_s, lower_value=low_v, upper_hue=high_h, upper_saturation=high_s, upper_value=high_v,sources = source) 
        '''
         if bgr: #BGR filter(Background white)
            dataset = LoadStreams(img_size=imgsz, stride=stride, img_filter=img_filter, lower_red=low_r, lower_green=low_g, lower_blue=low_b, upper_red=high_r, upper_green=high_g, upper_blue=high_b, lower_hue=low_h, lower_saturation=low_s, lower_value=low_v, upper_hue=high_h, upper_saturation=high_s, upper_value=high_v,sources = source) 
        else:   #HSV filter(Background black)
            dataset = LoadStreams(sources=source, img_size=imgsz, stride=stride, filter=bgr, lower_red = low_r, lower_green=low_g, lower_blue=low_b, upper_red=high_r, upper_green=high_g, upper_blue=high_b)  #multiple input for lower_red
        #Cannot open streams.txt
        '''
       
        #dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        #dataset = LoadStreams(source, img_size=imgsz, stride=stride, hue=h, saturation=s, value=v)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, img_filter=img_filter, lower_red = low_r, lower_green=low_g, lower_blue=low_b, upper_red=high_r, upper_green=high_g, upper_blue=high_b, lower_hue = low_h, lower_saturation=low_s, lower_value=low_v, upper_hue=high_h, upper_saturation=high_s, upper_value=high_v)
        '''
        if bgr:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, filter=bgr, lower_red = low_r, lower_green=low_g, lower_blue=low_b, upper_red=high_r, upper_green=high_g, upper_blue=high_b)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, filter=bgr, lower_hue = low_h, lower_saturation=low_s, lower_value=low_v, upper_hue=high_h, upper_saturation=high_s, upper_value=high_v)
        '''
        #dataset = LoadImages(source, img_size=imgsz, stride=stride, hue=h, saturation=s, value=v)
        #dataset = LoadImages(source, img_size=imgsz, stride=stride, hue=hue, saturation=saturation, value=value)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        #img = filter(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                '''
                Det size (baris x kolom) = jumlah obyek yg terdeteksi x 6(1-4:coordinate BBox, 5 = confidence, 6 = class)
                
                z = det.shape
                y = det.size()
                print(f'{z}Shape of det')
                print(f'{y}Size of det')
                print(f'{det}Det value')
                '''
                # Print results
                for c in det[:, -1].unique():
                    #c = tuple berisi daftar kelas (cuma muncul sekali)
                    n = (det[:, -1] == c).sum()  # detections per class
                    #n = jumlah dari kelas c (misal kelas 0 ada 2 obyek yg terdeteksi brti n=2)

                    #Selama masih ada double detection
                    while n>1:
                        #Bikin algoritma buat milih yg paling tinggi confidencenya 
                        #Ambil row terkecildari kolom -2
                        min_row_index = torch.argmin(det[:,-2]).item() #// (det[:,-2]).size(0)
                        print(f'{min_row_index}min row value')
                        #Hapus row terkecil
                        det = torch.cat((det[:min_row_index], det[min_row_index+1:]))
                        #pass
                        print(f'{det}Det value')
                        n -= 1

                        #Detect angka 3
                        if names[int(c)] == "3" and names[int(c)] != "Down":
                            #BOOLEAN TRUE
                            third_floor = True
                        
                        #Detect kalo ada yajirushi
                        if names[int(c)] == "Up" or names[int(c)] == "Down":
                            #BOOLEAN TRUE
                            yajirushi = True

                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                #Detect if arrive at 3rd Floor(Ada angka 3 tapi yajirushi gaada)
                if third_floor and not yajirushi:
                    #Send Signal to PLC
                    print(f'Arrived at 3rd Floor')

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/v5lite-s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='sample', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--low_hue', help='Lower Hue value', default=0)
    parser.add_argument('--low_saturation', help='Lower Saturation value', default=0)
    parser.add_argument('--low_value', help='Lower Value value', default=0)
    parser.add_argument('--high_hue', help='Lower Hue value', default=179)
    parser.add_argument('--high_saturation', help='Lower Saturation value', default=255)
    parser.add_argument('--high_value', help='Lower Value value', default=255)
    parser.add_argument('--low_red', help='Lower Hue value', default=0)
    parser.add_argument('--low_green', help='Lower Saturation value', default=0)
    parser.add_argument('--low_blue', help='Lower Value value', default=0)
    parser.add_argument('--high_red', help='Lower Hue value', default=255)
    parser.add_argument('--high_green', help='Lower Saturation value', default=255)
    parser.add_argument('--high_blue', help='Lower Value value', default=255)
    parser.add_argument('--img_filter',type=str, default='no', help='no or BGR or HSV')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()

"""
Update Log
・20240528: Line 181 Add Signal Send Controller to PLC when arrived at 3rd floor
"""
