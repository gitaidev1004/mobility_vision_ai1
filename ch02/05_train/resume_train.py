# -*- coding: utf-8 -*-
"""중단된 학습을 체크포인트에서 재개한다.

CPU 로 100 에폭을 돌리면 수 시간이 걸린다. 그 사이 정전, 절전 모드 진입,
터미널 종료로 학습이 끊기는 일이 실제로 자주 발생한다.
Ultralytics 는 last.pt 에 옵티마이저 상태와 에폭 번호까지 저장하므로
`resume=True` 로 중단 지점부터 이어서 학습할 수 있다.

주의
  - 재개는 '같은 설정'으로만 가능하다. 에폭 수나 배치 크기를 바꾸려면
    재개가 아니라 best.pt 를 시작 가중치로 둔 새 학습을 해야 한다.
  - best.pt 가 아니라 last.pt 로 재개한다. best.pt 에는 옵티마이저
    상태가 없어 학습률 스케줄이 처음부터 다시 시작된다.

사용법:
    python 05_train/resume_train.py --ckpt runs/plate/exp/weights/last.pt
    python 05_train/resume_train.py --ckpt runs/plate/exp/weights/best.pt --new-run --epochs 40
"""
from __future__ import annotations

import argparse
import pathlib
import sys


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="학습 재개")
    ap.add_argument("--ckpt", default="runs/plate/exp/weights/last.pt",
                    help="체크포인트 경로. 재개는 last.pt 를 쓴다")
    ap.add_argument("--new-run", action="store_true",
                    help="재개가 아니라 이 가중치를 시작점으로 새 학습을 한다")
    ap.add_argument("--data", default="05_train/data.yaml")
    ap.add_argument("--epochs", type=int, default=40,
                    help="--new-run 일 때만 쓰인다")
    ap.add_argument("--lr0", type=float, default=0.002,
                    help="--new-run 일 때. 이어 학습은 더 낮은 학습률로 시작한다")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--project", default="runs/plate")
    ap.add_argument("--name", default="exp_resume")
    return ap.parse_args()


def list_checkpoints(root: pathlib.Path) -> None:
    """runs 디렉터리에서 사용 가능한 체크포인트를 찾아 보여준다."""
    if not root.is_dir():
        print(f"  (디렉터리 없음: {root})")
        return
    found = sorted(root.rglob("*.pt"))
    if not found:
        print(f"  (체크포인트 없음: {root})")
        return
    for p in found:
        size = p.stat().st_size / (1024 * 1024)
        print(f"  {p}  ({size:.1f} MB)")


def main() -> int:
    args = parse_args()
    ckpt = pathlib.Path(args.ckpt)

    try:
        from ultralytics import YOLO            # noqa: PLC0415
    except ImportError:
        print("[안내] ultralytics 가 없다. pip install ultralytics==8.3.200")
        print("\n현재 보이는 체크포인트 목록:")
        list_checkpoints(pathlib.Path(args.project))
        return 0

    if not ckpt.exists():
        print(f"[안내] 체크포인트가 없다: {ckpt}")
        print("\n사용 가능한 체크포인트 목록:")
        list_checkpoints(pathlib.Path(args.project))
        print("\n먼저 05_train/train_yolo.py 로 학습을 시작한다.")
        return 0

    size_mb = ckpt.stat().st_size / (1024 * 1024)
    print("=== 체크포인트 ===")
    print(f"  경로 : {ckpt}")
    print(f"  크기 : {size_mb:.1f} MB")

    model = YOLO(str(ckpt))

    if args.new_run:
        # best.pt 를 출발점으로 삼아 더 낮은 학습률로 추가 학습한다.
        # 데이터를 보강한 뒤 기존 성능을 유지하며 개선할 때 쓰는 방식이다.
        print(f"\n[새 학습] 시작 가중치로 사용. epochs={args.epochs} lr0={args.lr0}")
        model.train(
            data=args.data, epochs=args.epochs, lr0=args.lr0,
            imgsz=args.imgsz, batch=args.batch, device=args.device,
            project=args.project, name=args.name, exist_ok=True,
            warmup_epochs=1.0,     # 이미 학습된 가중치이므로 워밍업을 짧게 한다
            verbose=True,
        )
    else:
        # resume=True 는 last.pt 에 저장된 에폭/옵티마이저/스케줄러를 복원한다.
        # 이때 data/epochs 등 인자는 체크포인트에 기록된 값이 쓰이므로
        # 따로 넘기지 않는다(넘겨도 무시되거나 충돌 경고가 난다).
        print("\n[재개] last.pt 의 에폭/옵티마이저 상태를 복원해 이어서 학습한다.")
        model.train(resume=True)

    print("\n=== 완료 ===")
    print("  결과 디렉터리의 results.csv 에서 에폭 번호가 이어졌는지 확인한다.")
    print("  이어지지 않았다면 best.pt 로 재개했을 가능성이 높다. last.pt 를 쓴다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
