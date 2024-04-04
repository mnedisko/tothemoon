import cv2
import numpy as np
from ultralytics import YOLO

settings={"moon_path":"/home/oem/İndirilenler/fullmoon.png",
          "video_path":"moon2.mp4",
          "model_path":"fullmoondetectmodel.pt",
          "cls_model_path":"runs/classify/train11/weights/last.pt",
          "video_saving":True}

overlay = cv2.imread(settings["moon_path"], cv2.IMREAD_UNCHANGED)
overlay=cv2.resize(overlay,(100,100))  # IMREAD_UNCHANGED => open image with the alpha channel
alpha = np.ones((overlay.shape[0], overlay.shape[1]), dtype=np.uint8) * 255
overlay = cv2.merge((overlay, alpha))

def resize_moon(overlay):
    """
    Resize the overlay image to match the size of the bounding box.

    Args:
        overlay (numpy.ndarray): The overlay image.

    Returns:
        numpy.ndarray: The resized overlay image.
    """
    overlay_height, overlay_width = overlay.shape[:2]
    width_ratio = bbox_width / overlay_width
    height_ratio = bbox_height / overlay_height
    overlay = cv2.resize(overlay, (int(overlay_width * width_ratio), int(overlay_height * height_ratio)))
    return overlay

def get_alpha(overlay, midx, midy, frame):
    """
    Apply alpha blending to overlay the resized moon image onto the frame.

    Args:
        overlay (numpy.ndarray): The resized moon image with alpha channel.
        midx (float): The x-coordinate of the midpoint of the bounding box.
        midy (float): The y-coordinate of the midpoint of the bounding box.
        frame (numpy.ndarray): The frame to overlay the moon image on.

    Returns:
        numpy.ndarray: The frame with the moon image overlaid.
    """
    overlay=resize_moon(overlay)
    alpha_channel = overlay[:, :, 3] / 255
    overlay_colors = overlay[:, :, :3]
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))
    h, w = overlay.shape[:2]
    x1 = int(midx - w / 2)
    y1 = int(midy - h / 2)
    x2 = x1 + w
    y2 = y1 + h
    x1_frame, y1_frame = max(0, x1), max(0, y1)
    x2_frame, y2_frame = min(frame.shape[1], x2), min(frame.shape[0], y2)
    moon_x1 = x1_frame - x1
    moon_x2 = moon_x1 + (x2_frame - x1_frame)
    moon_y1 = y1_frame - y1
    moon_y2 = moon_y1 + (y2_frame - y1_frame)

    background_subsection = frame[y1_frame:y2_frame, x1_frame:x2_frame]
    composite = background_subsection * (1 - alpha_mask[moon_y1:moon_y2, moon_x1:moon_x2]) + overlay_colors[moon_y1:moon_y2, moon_x1:moon_x2] * alpha_mask[moon_y1:moon_y2, moon_x1:moon_x2]
    frame[y1_frame:y2_frame, x1_frame:x2_frame] = composite

    return frame

cap = cv2.VideoCapture(settings["video_path"])
if settings["video_saving"]:
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'XVID'), 30, (int(cap.get(3)), int(cap.get(4))))
model = YOLO(settings["model_path"])
cls_model = YOLO(settings["cls_model_path"])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_Cpy= frame.copy()
    results = model(frame, show=False,verbose=False)
    cls_results = cls_model(frame, show=False,verbose=False)
    for boxes in results[0].boxes.cpu().numpy():
        x1, y1, x2, y2= boxes.xyxy[0]
        bbox_x, bbox_y, bbox_width, bbox_height = x1, y1, x2-x1, y2-y1
        cls = boxes.cls[0]
        conf = boxes.conf[0]
        mid_point = (x1+x2)/2, (y1+y2)/2
        midx = mid_point[0]
        midy = mid_point[1]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        frame = get_alpha(overlay,midx,midy, frame_Cpy)
    for result in cls_results:
        name = result.probs.top1
        if name == 0:
            name = 'no_zoom'
        else:
            name = 'zoom'
        cls = result.probs.top1conf
        cv2.putText(frame, f'{name} {cls:.2f}', (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    if settings["video_saving"]:
        out.write(frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
if settings["video_saving"]:
    out.release()

cap.release()
cv2.destroyAllWindows()
