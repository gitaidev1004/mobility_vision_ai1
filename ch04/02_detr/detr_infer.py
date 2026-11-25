# -*- coding: utf-8 -*-
"""HuggingFace transformers 의 DETR 추론 예시.

transformers / torch 가 없거나 가중치를 받을 수 없으면 안내를 출력하고 정상 종료한다.
사용법:
    python detr_infer.py                 (테스트 이미지 생성)
    python detr_infer.py car.jpg         (실제 이미지)
"""
import sys
import time

import numpy as np

MODEL_ID = "facebook/detr-resnet-50"


def make_sample_image(h=480, w=640):
    """차량과 번호판을 단순화한 테스트 이미지를 만든다."""
    img = np.full((h, w, 3), 120, dtype=np.uint8)
    img[260:420, 140:520] = (60, 70, 90)        # 차체
    img[352:384, 270:400] = (235, 235, 230)     # 번호판 바탕
    for x in range(278, 396, 14):
        img[358:378, x:x + 5] = (25, 25, 30)    # 문자 흉내
    return img


def check_deps():
    """필요한 패키지를 점검한다. 없으면 안내 문자열을 반환한다."""
    missing = []
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    try:
        import transformers  # noqa: F401
    except ImportError:
        missing.append("transformers")
    try:
        import PIL  # noqa: F401
    except ImportError:
        missing.append("Pillow")
    return missing


def main():
    missing = check_deps()
    if missing:
        print("[안내] DETR 추론에 필요한 패키지가 없다:", ", ".join(missing))
        print("       아래 명령으로 설치한 뒤 다시 실행한다.")
        print("       pip install torch --index-url "
              "https://download.pytorch.org/whl/cpu")
        print("       pip install transformers pillow timm")
        print("[안내] 패키지가 없어도 이 예제는 오류 없이 종료한다.")
        return 0

    import torch
    from PIL import Image
    from transformers import DetrForObjectDetection, DetrImageProcessor

    # 이미지 준비
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

    # 모델 로드. 가중치를 받을 수 없으면 안내 후 종료한다
    try:
        processor = DetrImageProcessor.from_pretrained(MODEL_ID)
        model = DetrForObjectDetection.from_pretrained(MODEL_ID)
    except Exception as exc:                            # noqa: BLE001
        print(f"[안내] 사전학습 가중치를 불러올 수 없다: {type(exc).__name__}")
        print(f"       인터넷이 연결된 환경에서 '{MODEL_ID}' 를 한 번 내려받아야 한다.")
        print("       오프라인이라면 HF_HOME 캐시 디렉터리를 미리 복사해 둔다.")
        return 0

    model.eval()
    torch.set_num_threads(4)        # CPU 추론 스레드 수를 명시한다

    inputs = processor(images=image, return_tensors="pt")
    print(f"[전처리] pixel_values shape = {tuple(inputs['pixel_values'].shape)}")

    # 추론 시간 측정. 첫 회는 준비 과정이 섞이므로 따로 본다
    with torch.no_grad():
        t0 = time.perf_counter()
        outputs = model(**inputs)
        t_first = time.perf_counter() - t0

        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            model(**inputs)
            times.append(time.perf_counter() - t0)

    print(f"[추론] logits shape = {tuple(outputs.logits.shape)} "
          "(배치, 쿼리 100, 클래스 91+1)")
    print(f"[추론] pred_boxes shape = {tuple(outputs.pred_boxes.shape)} "
          "(배치, 쿼리 100, cxcywh)")
    print(f"[시간] 첫 추론 {t_first * 1000:.1f} ms, "
          f"이후 평균 {np.mean(times) * 1000:.1f} ms")

    # 후처리: 임계값 이상인 결과만 남긴다. DETR 은 NMS 가 필요하지 않다
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

    print("\n[참고] 이 가중치는 COCO 80 클래스로 학습되어 '번호판' 클래스가 없다.")
    print("       번호판 검출에 쓰려면 단일 클래스 데이터셋으로 미세 조정해야 한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
