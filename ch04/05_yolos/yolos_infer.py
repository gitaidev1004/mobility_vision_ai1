# -*- coding: utf-8 -*-
"""HuggingFace transformers 의 YOLOS 추론 예시.

transformers / torch 가 없거나 가중치를 받을 수 없으면 안내 후 정상 종료한다.
사용법:
    python yolos_infer.py                (테스트 이미지 생성)
    python yolos_infer.py car.jpg        (실제 이미지)
"""
import sys
import time

import numpy as np

MODEL_ID = "hustvl/yolos-tiny"


def make_sample_image(h=480, w=640):
    img = np.full((h, w, 3), 120, dtype=np.uint8)
    img[260:420, 140:520] = (60, 70, 90)
    img[352:384, 270:400] = (235, 235, 230)
    for x in range(278, 396, 14):
        img[358:378, x:x + 5] = (25, 25, 30)
    return img


def check_deps():
    missing = []
    for mod, pkg in (("torch", "torch"), ("transformers", "transformers"),
                     ("PIL", "Pillow")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    return missing


def main():
    missing = check_deps()
    if missing:
        print("[안내] YOLOS 추론에 필요한 패키지가 없다:", ", ".join(missing))
        print("       pip install torch --index-url "
              "https://download.pytorch.org/whl/cpu")
        print("       pip install transformers pillow")
        print("[안내] 패키지가 없어도 이 예제는 오류 없이 종료한다.")
        return 0

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    if len(sys.argv) > 1:
        try:
            image = Image.open(sys.argv[1]).convert("RGB")
            print(f"[입력] {sys.argv[1]}  크기 {image.size}")
        except OSError as exc:
            print(f"[안내] 이미지를 열 수 없다: {exc}. 테스트 이미지로 대체한다.")
            image = Image.fromarray(make_sample_image())
    else:
        image = Image.fromarray(make_sample_image())
        print("[안내] 인자가 없어 640x480 테스트 이미지를 생성했다.")

    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForObjectDetection.from_pretrained(MODEL_ID)
    except Exception as exc:                            # noqa: BLE001
        print(f"[안내] 사전학습 가중치를 불러올 수 없다: {type(exc).__name__}")
        print(f"       인터넷이 연결된 환경에서 '{MODEL_ID}' 를 한 번 내려받아야 한다.")
        return 0

    model.eval()
    torch.set_num_threads(4)

    inputs = processor(images=image, return_tensors="pt")
    pv = inputs["pixel_values"]
    print(f"[전처리] pixel_values shape = {tuple(pv.shape)}")

    # YOLOS 는 패치 토큰 수가 입력 크기로 결정된다
    patch = model.config.patch_size
    gh, gw = pv.shape[2] // patch, pv.shape[3] // patch
    n_det = model.config.num_detection_tokens
    print(f"[토큰] 패치 {patch}px -> 격자 {gh}x{gw} = {gh * gw} 패치 토큰")
    print(f"[토큰] 검출 토큰 {n_det} 개, 전체 시퀀스 {gh * gw + n_det} 개")
    print(f"[설정] hidden_size={model.config.hidden_size}, "
          f"layers={model.config.num_hidden_layers}, "
          f"heads={model.config.num_attention_heads}")

    with torch.no_grad():
        t0 = time.perf_counter()
        outputs = model(**inputs)
        t_first = time.perf_counter() - t0
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            model(**inputs)
            times.append(time.perf_counter() - t0)

    print(f"[추론] logits shape = {tuple(outputs.logits.shape)}")
    print(f"[추론] pred_boxes shape = {tuple(outputs.pred_boxes.shape)}")
    print(f"[시간] 첫 추론 {t_first * 1000:.1f} ms, "
          f"이후 평균 {np.mean(times) * 1000:.1f} ms")

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.5)[0]
    print(f"\n[검출 결과] score >= 0.5 인 객체 {len(results['scores'])} 개")
    for score, label, box in zip(results["scores"], results["labels"],
                                 results["boxes"]):
        name = model.config.id2label[int(label)]
        x1, y1, x2, y2 = [round(float(v), 1) for v in box]
        print(f"  {name:<14s} score={float(score):.3f} "
              f"box=({x1}, {y1}, {x2}, {y2})")

    print("\n[참고] 같은 이미지를 입력 크기만 바꿔 넣으면 지연시간이 어떻게 변하는지")
    print("       반드시 측정해 보아야 한다. 토큰 수가 2 배면 어텐션 비용은 4 배다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
