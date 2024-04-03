import os
import cv2
nozoompoint = []
zoompoint = []
def mouse(event, x, y, flags, param):
    if event == cv2.EVENT_MBUTTONDOWN:
        print(x, y)
        nozoompoint.append((x, y))
        print(nozoompoint)
    if event == cv2.EVENT_RBUTTONDOWN:
        print(x, y)
        zoompoint.append((x, y))
        print(zoompoint)

impath = 'im0'
outpath = 'im1'
zoomoutpath = "im2"
if not os.path.exists(zoomoutpath):
    os.makedirs(zoomoutpath)
if not os.path.exists(outpath):
    os.makedirs(outpath)
for filename in os.listdir(impath):
    frame = cv2.imread(os.path.join(impath, filename))
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', mouse)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        print( nozoompoint[0][0])
        cropped_frame = frame[nozoompoint[0][1]-100:nozoompoint[0][1]+100, nozoompoint[0][0]-100:nozoompoint[0][0]+100]
        cv2.imshow('cframe', cropped_frame)
        cv2.imwrite(os.path.join(outpath, filename), cropped_frame)
        nozoompoint.clear()
    if key ==ord('w'):
        print( nozoompoint[0][0])
        cropped_frame = frame[nozoompoint[0][1]-100:nozoompoint[0][1]+100, nozoompoint[0][0]-100:nozoompoint[0][0]+100]
        cv2.imshow('cframe', cropped_frame)
        cv2.imwrite(os.path.join(zoomoutpath, filename), cropped_frame)
        nozoompoint.clear()



