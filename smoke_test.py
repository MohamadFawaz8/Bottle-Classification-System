import numpy as np
from pipeline import run_pipeline
from counter import VideoCounter, WebcamCounter

img = np.zeros((480, 640, 3), dtype=np.uint8)
vc = VideoCounter()
wb = WebcamCounter(frame_width=640)

annotated, text, dets = run_pipeline(img, vc)
print('pipeline returns', type(annotated), len(text), dets)

for x in [100, 300, 500]:
    det = {'box': [x-20, 200, x+20, 260]}
    wb.update([det])
print('webcam count', wb.get_count())
