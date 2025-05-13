# 🌕 To The Moon

This is a fun little computer vision project that detects objects in a video stream and adds a moon image around each of them. Inspired by the idea of making even the most ordinary scenes feel a bit more cosmic 🌌

## 🚀 Getting Started

Make sure you have Python 3.6+ installed.

Then install the required dependencies:

```bash
pip install opencv-python numpy torch torchvision ultralytics
```

## 🎬 How to Use

Just run:

```bash
python main.py
```

* It will open your camera or video stream.
* Detected objects will be surrounded by a moon image.
* Press `q` to quit.
* The output video will be saved as `output.mp4`.

## ✨ Example Outputs

Here’s what the project looks like in action:

![Sample 1](img/original.gif) <!-- MARK: GIF 1 -->

![Sample 2](img/output.gif) <!-- MARK: GIF 2 -->

## 📁 Project Structure

```
├── main.py              # Main script
├── crop.py              # Cropping logic
├── laber.py             # Label-related logic
├── fullmoondetectmodel.pt  # The detection model
└── img/
    ├── moon.png         # Moon overlay image
    ├── sample1.gif
    └── sample2.gif
```

## 📌 Notes

* The detection model is YOLO-based.
* Moon position is calculated based on the bounding box of detected objects.
* Can be easily adapted to use different overlays (e.g. sun, stars, etc.).

## 📝 License

MIT — do whatever you want, just don't forget to have fun 🌝

---

If you liked this, feel free to fork it or drop a 🌟
