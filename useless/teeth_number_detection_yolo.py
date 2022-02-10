from yolov5.train import run

class teeth_number_detection_yolo():
    def __init__(self):
        run(data='custom-coco.yaml', imgsz=640, weights='./yolov5s.pt', batch=16, epochs = 100)
        # run(data='custom-coco-aihub.yaml', imgsz=640, weights='./yolov5s.pt', batch=16, epochs = 3)
        # python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt --cache
