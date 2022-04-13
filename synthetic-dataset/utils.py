import cv2


def read_video(file):
    # Read smoke video
    cap = cv2.VideoCapture(file)
    imgs = []
    ret = True
    while(cap.isOpened() and ret):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            imgs.append(frame)
     
    return imgs
