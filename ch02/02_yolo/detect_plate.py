# -*- coding: utf-8 -*-
"""Ultralytics YOLO 로 번호판을 검출하는 최소 스크립트.

ultralytics 패키지나 가중치 파일이 없으면 오류로 멈추지 않고
설치/준비 방법을 안내한 뒤 정상 종료한다(종료 코드 0).

사용법:
    python 02_yolo/detect_plate.py --weights weights/plate_yolo.pt --source samples/car01.jpg
"""
from __future__ import annotations

import argparse
import pathlib
import sys

GUIDE = """\
[안내] Ultralytics YOLO 를 사용할 수 없다.
  1) 가상환경 활성화 후 설치:  pip install ultralytics==8.3.200
  2) 학습된 번호판 가중치를 weights/plate_yolo.pt 에 둔다.
     아직 학습하지 않았다면 2.5절의 train_yolo.py 를 먼저 실행한다.
  3) 가중치가 없는 동안에는 2.1절 iou_demo.py / nms_demo.py 로
     검출 결과 후처리 원리를 먼저 익힐 수 있다.
"""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="YOLO 번호판 검출")
    ap.add_argument("--weights", default="weights/plate_yolo.pt",
                    help="학습된 가중치 경로(.pt)")
    ap.add_argument("--source", default="samples/car01.jpg",
                    help="이미지 파일 또는 디렉터리")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="추론 입력 해상도(정사각 기준 한 변)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="신뢰도 임계값. 낮추면 재현율↑ 정밀도↓")
    ap.add_argument("--iou", type=float, default=0.45,
                    help="NMS IoU 임계값")
    ap.add_argument("--save-dir", default="runs/detect_plate",
                    help="시각화 결과 저장 디렉터리")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from ultralytics import YOLO          # noqa: PLC0415
    except ImportError:
        print(GUIDE)
        return 0

    wpath = pathlib.Path(args.weights)
    if not wpath.exists():
        print(f"[안내] 가중치 파일이 없다: {wpath}")
        print(GUIDE)
        return 0

    spath = pathlib.Path(args.source)
    if not spath.exists():
        print(f"[안내] 입력 이미지가 없다: {spath}")
        print("      samples/ 디렉터리에 차량 사진을 넣고 다시 실행한다.")
        return 0

    # 모델을 메모리에 올린다. CPU 환경에서는 device='cpu' 로 고정하는 편이 안전하다
    model = YOLO(str(wpath))

    results = model.predict(
        source=str(spath),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device="cpu",
        verbose=False,
        save=True,
        project=args.save_dir,
        name="exp",
        exist_ok=True,
    )

    total = 0
    for r in results:
        img_name = pathlib.Path(getattr(r, "path", "unknown")).name
        boxes = r.boxes
        n = 0 if boxes is None else len(boxes)
        total += n
        print(f"\n[{img_name}] 검출 {n} 건  (원본 {r.orig_shape[1]}x{r.orig_shape[0]})")
        if n == 0:
            print("  번호판을 찾지 못했다. conf 를 낮추거나 imgsz 를 키워 재시도한다.")
            continue
        for i in range(n):
            x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].tolist()]
            score = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])
            name = model.names.get(cls_id, str(cls_id))
            w, h = x2 - x1, y2 - y1
            ratio = w / h if h else 0.0
            print(f"  #{i} label={name} score={score:.3f} "
                  f"bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) "
                  f"w={w:.0f}px h={h:.0f}px 종횡비={ratio:.2f}")
            # 번호판 폭이 너무 작으면 OCR 단계에서 판독이 어렵다
            if w < 120:
                print("     경고: 번호판 폭 120px 미만 - 재촬영 또는 확대 촬영 권장")

        # 추론 단계별 소요시간(ms). 전처리/추론/후처리로 나뉘어 보고된다
        sp = r.speed
        print(f"  소요시간 preprocess={sp['preprocess']:.1f}ms "
              f"inference={sp['inference']:.1f}ms "
              f"postprocess={sp['postprocess']:.1f}ms")

    print(f"\n총 검출 {total} 건. 시각화 결과는 {args.save_dir}/exp 에 저장되었다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
