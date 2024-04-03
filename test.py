from ultralytics import YOLO
import time
import cv2
import numpy as np
model = YOLO('fullmoondetectmodel.pt')
midx = 0
midy = 0
cls_model = YOLO('runs/classify/train11/weights/best.pt')
#results = model('moon2.mp4',show=True)

moon = cv2.imread('img/ay.png')
alpha = np.ones((moon.shape[0], moon.shape[1]), dtype=np.uint8) * 255
moon = cv2.merge((moon, alpha))
#def is_zoomed(frame, midx, midy, moon):
#    overlay_img = np.ones(frame.shape,np.uint8) * 255
#    rows, cols, _ = frame.shape
#    moon_rows, moon_cols, _ = moon.shape
#    overlay_img[int(midx - moon_rows/2):int(midx - moon_rows/2) + moon_rows, 
#                int(midy - moon_cols/2):int(midy - moon_cols/2) + moon_cols] = moon
#    moongray = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
#    ret, mask = cv2.threshold(moongray, 10, 255, cv2.THRESH_BINARY)
#    mask_inv = cv2.bitwise_not(mask)
#    moon_fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
#    frame_bg = cv2.bitwise_and(overlay_img, overlay_img, mask=mask)
#    dst = cv2.add(moon_fg, frame_bg)
#    return dst
def is_zoomed(frame, midx, midy, moon):
    rows, cols, _ = moon.shape
    x1 = int(midx - cols / 2)
    y1 = int(midy - rows / 2)
    x2 = x1 + cols
    y2 = y1 + rows

    # Arka plan görüntüsünü ve üst üste binen görüntüyü alfa kanalıyla birleştirme
    for y in range(y1, y2):
        for x in range(x1, x2):
            if x < 0 or y < 0 or x >= frame.shape[1] or y >= frame.shape[0]:
                continue  # Arka plan sınırlarını aşarsa devam et
            alpha_moon = moon[y - y1, x - x1, 3] / 255.0  # Alfa kanalını al
            alpha_frame = 1.0 - alpha_moon  # Arka plan alfa kanalı

            # Alfa kanalı ile ağırlıklandırılmış renkleri birleştirme
            for c in range(0, 3):
                frame[y, x, c] = (alpha_moon * moon[y - y1, x - x1, c] +
                                  alpha_frame * frame[y, x, c])
    return frame


#def is_zoomed(frame,midx,midy):
#    overlay_img = np.ones((frame.shape), np.uint8) * 255
#    overlay_img[int(midx):rows+int(midx), int(midy):cols+int(midy)] = moon
#    moongray = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
#    ret, mask = cv2.threshold(moongray, 10, 255, cv2.THRESH_BINARY)
#    mask_inv = cv2.bitwise_not(mask)
#    moon_fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
#    frame_bg = cv2.bitwise_and(overlay_img, overlay_img, mask=mask)
#    dst = cv2.add(moon_fg,frame_bg)
#    frame[int(midx):rows+int(midx), int(midy):cols+int(midy)] = moon
#    return frame
cap = cv2.VideoCapture('/home/oem/output.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_cpy = frame.copy()
    results = model(frame)
    cls_results = cls_model(frame)
    for boxes in results[0].boxes.cpu().numpy():
        x1, y1, x2, y2= boxes.xyxy[0]
        cls = boxes.cls[0]
        conf = boxes.conf[0]
        mid_point = (x1+x2)/2, (y1+y2)/2
        midx = mid_point[0]
        midy = mid_point[1]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (int((x1+x2)/2), int((y1+y2)/2)), 5, (0, 255, 0), -1)

    for result in cls_results:
        name = result.probs.top1
        if name == 0:
            name = 'no_zoom'
        else:
            name = 'zoom'
            frame = is_zoomed(frame_cpy, midx, midy, moon)
        cls = result.probs.top1conf
        cv2.putText(frame, f'{name} {cls:.2f}', (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            

    cv2.imshow('frame', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()