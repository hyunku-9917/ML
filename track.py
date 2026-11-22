# 학습한 YOLO 모델로 영상 속 객체를 추적하고 이동 궤적을 그려 저장합니다.
# 사용 예: python track.py --weights weights/yolov8n_best.pt --source my_video.mp4 --tracker bytetrack.yaml --seconds 5
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--start", type=float, default=0.0, help="시작 시점(초)")
    ap.add_argument("--seconds", type=float, default=5.0, help="처리할 길이(초)")
    ap.add_argument("--trail", type=int, default=30, help="궤적 길이(프레임)")
    ap.add_argument("--out", default="track_out")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(a.weights)
    cap = cv2.VideoCapture(a.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start, n = int(round(a.start * fps)), int(round(a.seconds * fps))
    hist, rows, vw, idx = defaultdict(list), [], None, -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx < start:
            continue
        if idx >= start + n:
            break
        r = model.track(frame, persist=True, tracker=a.tracker, verbose=False)[0]
        img = r.plot(color_mode="instance", line_width=2)
        if r.boxes is not None and r.boxes.is_track:
            for (cx, cy, _, _), tid, c in zip(r.boxes.xywh.cpu().numpy(), r.boxes.id.int().cpu().tolist(),
                                              r.boxes.cls.int().cpu().tolist()):
                hist[tid].append((int(cx), int(cy)))
                pts = np.array(hist[tid][-a.trail:], np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], False, colors(tid, True), 3)
                rows.append([idx - start, tid, model.names[c], round(float(cx), 1), round(float(cy), 1)])
        if vw is None:
            vw = cv2.VideoWriter(str(out / "track.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, img.shape[1::-1])
        vw.write(img)
    cap.release()
    if vw is not None:
        vw.release()
    with open(out / "trajectory.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["frame", "track_id", "class_name", "cx", "cy"])
        w.writerows(rows)
    print(f"saved: {out / 'track.mp4'}, {out / 'trajectory.csv'} (unique IDs: {len(hist)})")


if __name__ == "__main__":
    main()
