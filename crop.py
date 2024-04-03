import cv2
import os
cap = cv2.VideoCapture('moon2.mp4')
outpath = 'im0'
if not os.path.exists(outpath):
    os.makedirs(outpath)
count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    filename = os.path.join(outpath, f'frame_{count}.jpg')
    cv2.imwrite(filename, frame)
    print(f'Frame {count} saved')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
