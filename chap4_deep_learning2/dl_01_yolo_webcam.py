from ultralytics import YOLO


def detect(num):
    models = ['yolov8n.pt', 'yolov8n-pose.pt', 'yolov8n-seg.pt']
    model = YOLO(models[num])
    results = model(source=0, show=True)


def detect2():
    # plain
    model = YOLO('yolov8n.pt')
    results = model('https://ultralytics.com/images/bus.jpg', show=True)

    # pose
    model = YOLO('yolov8n-pose.pt')
    results = model('https://ultralytics.com/images/bus.jpg', show=True)

    # segmentation
    model = YOLO('yolov8n-seg.pt')
    results = model('https://ultralytics.com/images/bus.jpg', show=True)

    print('here')


def detect3():
    # Load a model
    model = YOLO('runs/detect/train/weights/best.pt')
    # load model 활용
    # results = model('Rock-Paper-Scissors-SXSW-11/test/images/egohands-public-1624465888641_png_jpg.rf.886253ff4aaf0b15969b9fa1a918c6de.jpg', show=True)

    # webcam 활용
    results = model(source=0, show=True)
    print('here')


if __name__ == '__main__':
    # detect(2)
    # detect2()
    detect3()
