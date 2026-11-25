# -*- coding: utf-8 -*-
"""번호판 검출 YOLO 학습 스크립트.

전이학습(Transfer Learning) 방식으로 COCO 사전학습 가중치에서 출발해
번호판 단일 클래스 데이터로 미세조정(fine-tuning)한다.

주요 기능
  - 하이퍼파라미터를 CLI 로 받는다(재현 실험을 위해 설정을 기록한다)
  - 학습 설정을 runs/.../train_config.json 에 저장한다
  - patience 로 EarlyStopping 을 적용한다
  - 학습 종료 후 best.pt / last.pt 위치와 검증 지표를 출력한다

ultralytics 가 없으면 설치 안내 후 정상 종료한다.

사용법:
    python 05_train/train_yolo.py --data 05_train/data.yaml --model yolo11n.pt \
        --epochs 100 --batch 16 --imgsz 640 --lr0 0.01 --patience 20
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

GUIDE = """\
[안내] Ultralytics YOLO 가 없어 학습을 진행할 수 없다.
  pip install ultralytics==8.3.200
설치 후 아래 순서로 준비한다.
  1) 03_dataset/split_dataset.py --copy 로 data/plate 를 만든다
  2) 04_label/label_lint.py 로 라벨 오류 0건을 확인한다
  3) 05_train/data.yaml 의 path 가 실제 경로를 가리키는지 확인한다
"""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="번호판 YOLO 학습")
    # --- 데이터/모델 ---
    ap.add_argument("--data", default="05_train/data.yaml",
                    help="데이터셋 yaml 경로")
    ap.add_argument("--model", default="yolo11n.pt",
                    help="시작 가중치. 사전학습 .pt 를 주면 전이학습이 된다")
    # --- 학습 길이 ---
    ap.add_argument("--epochs", type=int, default=100,
                    help="최대 에폭. patience 로 조기 종료될 수 있다")
    ap.add_argument("--patience", type=int, default=20,
                    help="EarlyStopping: 검증 성능이 이 에폭 수만큼 "
                         "개선되지 않으면 학습을 멈춘다")
    # --- 배치/해상도 ---
    ap.add_argument("--batch", type=int, default=16,
                    help="배치 크기. -1 을 주면 자동 산정(GPU 전용)")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="학습 입력 해상도. 소형 번호판이 많으면 키운다")
    ap.add_argument("--workers", type=int, default=4,
                    help="데이터 로더 프로세스 수. 윈도우는 0~4 를 권장한다")
    # --- 최적화 ---
    ap.add_argument("--optimizer", default="auto",
                    choices=["auto", "SGD", "Adam", "AdamW"])
    ap.add_argument("--lr0", type=float, default=0.01,
                    help="초기 학습률. 전이학습에서는 0.01 이하로 시작한다")
    ap.add_argument("--lrf", type=float, default=0.01,
                    help="최종 학습률 비율. lr0*lrf 까지 코사인으로 감쇠한다")
    ap.add_argument("--momentum", type=float, default=0.937)
    ap.add_argument("--weight-decay", type=float, default=0.0005)
    ap.add_argument("--warmup-epochs", type=float, default=3.0,
                    help="초기 몇 에폭 동안 학습률을 서서히 올린다")
    ap.add_argument("--freeze", type=int, default=0,
                    help="백본 앞쪽 N개 레이어를 동결한다. "
                         "데이터가 적을 때 10 정도를 시도한다")
    # --- 증강 ---
    ap.add_argument("--hsv-h", type=float, default=0.015)
    ap.add_argument("--hsv-s", type=float, default=0.7)
    ap.add_argument("--hsv-v", type=float, default=0.4,
                    help="명도 증강 폭. 야간 데이터가 적으면 키운다")
    ap.add_argument("--degrees", type=float, default=8.0,
                    help="회전 증강 각도. 기울어진 번호판 대응")
    ap.add_argument("--translate", type=float, default=0.10)
    ap.add_argument("--scale", type=float, default=0.5,
                    help="크기 증강. 원거리 소형 번호판 대응")
    ap.add_argument("--shear", type=float, default=2.0)
    ap.add_argument("--perspective", type=float, default=0.0005,
                    help="투시 증강. 측면 촬영 번호판 대응")
    ap.add_argument("--fliplr", type=float, default=0.0,
                    help="좌우 반전 확률. 번호판 문자가 뒤집히므로 0 으로 둔다")
    ap.add_argument("--mosaic", type=float, default=1.0,
                    help="모자이크 증강 확률")
    ap.add_argument("--close-mosaic", type=int, default=10,
                    help="마지막 N 에폭은 모자이크를 끈다(실제 분포에 맞춤)")
    # --- 환경/출력 ---
    ap.add_argument("--device", default="cpu",
                    help="'cpu' 또는 '0'(GPU 0번)")
    ap.add_argument("--project", default="runs/plate")
    ap.add_argument("--name", default="exp")
    ap.add_argument("--save-period", type=int, default=10,
                    help="N 에폭마다 체크포인트를 따로 남긴다(장시간 학습 보호)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def check_dataset(data_yaml: pathlib.Path) -> bool:
    """데이터 yaml 과 학습 이미지 디렉터리가 있는지 가볍게 확인한다."""
    if not data_yaml.exists():
        print(f"[안내] 데이터 yaml 이 없다: {data_yaml}")
        return False
    text = data_yaml.read_text(encoding="utf-8")
    if "names" not in text or "train" not in text:
        print(f"[안내] {data_yaml} 에 train/names 항목이 없다.")
        return False
    return True


def main() -> int:
    args = parse_args()

    try:
        from ultralytics import YOLO             # noqa: PLC0415
    except ImportError:
        print(GUIDE)
        return 0

    data_yaml = pathlib.Path(args.data)
    if not check_dataset(data_yaml):
        print(GUIDE)
        return 0

    out_dir = pathlib.Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    # 어떤 설정으로 얻은 성능인지 나중에 반드시 되짚게 된다. 먼저 기록해 둔다
    (out_dir / "train_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== 학습 설정 ===")
    for k in ("model", "data", "epochs", "batch", "imgsz", "lr0",
              "patience", "freeze", "device", "seed"):
        print(f"  {k:12s} = {getattr(args, k.replace('-', '_'))}")
    print(f"  설정 저장    = {out_dir / 'train_config.json'}")

    model = YOLO(args.model)        # 사전학습 가중치를 불러온다(전이학습 시작점)

    t0 = time.perf_counter()
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        patience=args.patience,          # EarlyStopping
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0, lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        freeze=args.freeze if args.freeze > 0 else None,
        hsv_h=args.hsv_h, hsv_s=args.hsv_s, hsv_v=args.hsv_v,
        degrees=args.degrees, translate=args.translate, scale=args.scale,
        shear=args.shear, perspective=args.perspective,
        fliplr=args.fliplr, mosaic=args.mosaic, close_mosaic=args.close_mosaic,
        device=args.device,
        project=args.project, name=args.name, exist_ok=True,
        save=True, save_period=args.save_period,
        plots=True, seed=args.seed, verbose=True,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n=== 학습 종료 ({elapsed / 60:.1f} 분) ===")
    print(f"  best 가중치 : {out_dir / 'weights' / 'best.pt'}")
    print(f"  last 가중치 : {out_dir / 'weights' / 'last.pt'}")
    print(f"  학습 곡선   : {out_dir / 'results.png'}")
    print(f"  지표 CSV    : {out_dir / 'results.csv'}")

    # 검증 세트로 한 번 더 평가해 요약 지표를 남긴다
    metrics = model.val(data=str(data_yaml), imgsz=args.imgsz,
                        device=args.device, verbose=False)
    box = metrics.box
    print("\n=== 검증 세트 지표 ===")
    print(f"  Precision    : {float(box.mp):.4f}")
    print(f"  Recall       : {float(box.mr):.4f}")
    print(f"  mAP@0.5      : {float(box.map50):.4f}")
    print(f"  mAP@0.5:0.95 : {float(box.map):.4f}")
    print("\n다음 단계: 06_eval/eval_detector.py 로 Test 세트를 평가한다.")
    print("※ 위 지표는 검증 세트 기준이다. 최종 보고에는 Test 세트 수치를 쓴다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
