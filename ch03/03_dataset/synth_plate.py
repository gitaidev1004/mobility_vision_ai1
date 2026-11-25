# -*- coding: utf-8 -*-
"""PIL 로 한국 번호판을 합성 생성한다(구형/신형 규격 + 증강).

실행:  python 03_dataset/synth_plate.py --count 24 --out data/synth
한글 폰트를 찾지 못하면 숫자만 그리고 경고를 출력한다(실행은 계속된다).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# 같은 폴더의 charset.py 를 어디서 실행해도 찾을 수 있게 경로를 명시한다.
sys.path.append(str(Path(__file__).resolve().parent))
from charset import PRIVATE_USE_CHARS, REGION_NAMES  # noqa: E402

#: 번호판 규격별 캔버스 크기와 글자 배치. 실제 규격(mm)의 비율을 반영한다.
#: 신형 1줄형 520x110mm(비율 4.7), 구형 2줄형 335x170mm(비율 2.0).
SPECS = {
    "new_1line": {"size": (520, 110), "rows": 1, "font_ratio": 0.62},
    "new_1line_3digit": {"size": (520, 110), "rows": 1, "font_ratio": 0.58},
    "old_2line": {"size": (335, 170), "rows": 2, "font_ratio": 0.42},
}

#: 번호판 배경색과 글자색(BGR 아님, RGB).
#: 전기차는 2017년부터 파란 바탕 번호판을 쓴다(계약서의 vehicle_type 과 연동된다).
PALETTES = {
    "private": {"bg": (248, 248, 246), "fg": (28, 28, 28)},    # 자가용 흰 바탕
    "commercial": {"bg": (242, 206, 60), "fg": (28, 28, 28)},  # 사업용 노란 바탕
    "electric": {"bg": (58, 110, 200), "fg": (246, 246, 246)}, # 전기차 파란 바탕
    "old_green": {"bg": (34, 120, 82), "fg": (246, 246, 246)}, # 구형 녹색 바탕
}

FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "C:/Windows/Fonts/malgunbd.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
)


def load_font(size: int) -> tuple[ImageFont.FreeTypeFont, bool]:
    """한글을 그릴 수 있는 폰트를 찾는다. 반환값 둘째는 '한글 가능' 여부다."""
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size), True
            except OSError:
                continue
    # 한글 폰트가 없으면 라틴 폰트로 대체한다(숫자만 정상 출력된다).
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size), False
    return ImageFont.load_default(), False


def random_plate_text(spec_name: str, rng: random.Random) -> tuple[str, str]:
    """규격에 맞는 번호판 문자열과 용도 분류를 무작위로 만든다.

    반환: (문자열, 팔레트 키)
    """
    use = rng.choice(PRIVATE_USE_CHARS)
    tail = "".join(str(rng.randrange(10)) for _ in range(4))
    if spec_name == "old_2line":
        region = rng.choice(REGION_NAMES)
        head = f"{rng.randrange(10)}{rng.randrange(10)}"
        return f"{region} {head}{use}{tail}", "old_green" if rng.random() < 0.3 else "private"
    if spec_name == "new_1line_3digit":
        head = "".join(str(rng.randrange(10)) for _ in range(3))
        # 3자리 확장형은 2019년 이후 신규 등록분이다. 전기차 비중을 높게 둔다.
        return f"{head}{use}{tail}", "electric" if rng.random() < 0.25 else "private"
    head = f"{rng.randrange(1, 10)}{rng.randrange(10)}"
    return f"{head}{use}{tail}", "private"


def render_plate(text: str, spec_name: str, palette_key: str,
                 letter_spacing: float = 0.0) -> Image.Image:
    """번호판 한 장을 깨끗하게 렌더링한다(증강 전 단계)."""
    spec = SPECS[spec_name]
    w, h = spec["size"]
    palette = PALETTES[palette_key]
    img = Image.new("RGB", (w, h), palette["bg"])
    draw = ImageDraw.Draw(img)
    # 번호판 테두리(실제 번호판은 검은 또는 흰 테두리가 있다)
    draw.rectangle([2, 2, w - 3, h - 3], outline=palette["fg"], width=3)

    if spec["rows"] == 2 and " " in text:
        region, number = text.split(" ", 1)
        lines = [region, number]
        ratios = [spec["font_ratio"] * 0.7, spec["font_ratio"] * 1.15]
    else:
        lines = [text.replace(" ", "")]
        ratios = [spec["font_ratio"]]

    # 글자가 테두리를 넘지 않도록 좌우 안쪽 여백을 둔다.
    usable_w = w * 0.88
    y_cursor = h * 0.08
    for line, ratio in zip(lines, ratios):
        # 폰트 크기를 먼저 정한 뒤, 줄 폭이 판을 넘으면 넘지 않을 때까지 줄인다.
        # 자동 축소가 없으면 '서울 34나5678' 처럼 글자 수가 많은 구형 번호판에서
        # 양 끝 글자가 판 밖으로 잘려 나가 라벨과 이미지가 어긋난다.
        size = max(8, int(h * ratio))
        while size > 8:
            font, _ = load_font(size)
            widths = [draw.textlength(ch, font=font) for ch in line]
            gap = font.size * letter_spacing
            total = sum(widths) + gap * max(0, len(line) - 1)
            if total <= usable_w:
                break
            size = int(size * 0.92)

        x = (w - total) / 2.0
        bbox = font.getbbox(line) if line else (0, 0, 0, 0)
        line_h = bbox[3] - bbox[1]
        y = y_cursor - bbox[1] if len(lines) == 2 else (h - line_h) / 2.0 - bbox[1]
        for ch, cw in zip(line, widths):
            draw.text((x, y), ch, font=font, fill=palette["fg"])
            x += cw + gap
        y_cursor += line_h + h * 0.06
    return img


def augment(img: Image.Image, rng: random.Random) -> tuple[Image.Image, dict]:
    """촬영 환경을 모사한 증강을 적용한다.

    증강을 설계할 때의 원칙: 실제로 관측되는 변형만 넣는다.
    번호판은 위아래로 뒤집히거나 좌우 반전되는 일이 없으므로 flip 은 쓰지 않는다.
    대신 원근, 회전, 밝기, 블러, 노이즈, JPEG 압축을 조합한다.
    """
    applied: dict = {}
    w, h = img.size

    # 1) 원근 왜곡 — 측면 진입 차량
    if rng.random() < 0.6:
        amp = rng.uniform(0.02, 0.10)
        applied["perspective"] = round(amp, 3)
        dx, dy = w * amp, h * amp
        # PIL 의 QUAD 변환은 '출력의 네 꼭짓점에 입력의 어디를 가져올지' 를 지정한다.
        quad = (
            rng.uniform(0, dx), rng.uniform(0, dy),
            rng.uniform(0, dx), h - rng.uniform(0, dy),
            w - rng.uniform(0, dx), h - rng.uniform(0, dy),
            w - rng.uniform(0, dx), rng.uniform(0, dy),
        )
        img = img.transform((w, h), Image.QUAD, quad, resample=Image.BICUBIC)

    # 2) 회전 — 카메라 설치 각도와 노면 경사
    if rng.random() < 0.7:
        angle = rng.uniform(-6.0, 6.0)
        applied["rotate_deg"] = round(angle, 2)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False,
                         fillcolor=(70, 70, 70))

    arr = np.asarray(img, dtype=np.float32)

    # 3) 밝기/대비 — 주야간, 역광
    gain = rng.uniform(0.35, 1.25)
    bias = rng.uniform(-25.0, 35.0)
    applied["gain"] = round(gain, 3)
    applied["bias"] = round(bias, 1)
    arr = arr * gain + bias

    # 4) 노이즈 — 야간 고감도 촬영
    sigma = rng.uniform(2.0, 22.0)
    applied["noise_sigma"] = round(sigma, 1)
    arr = arr + np.random.default_rng(rng.randrange(1 << 30)).normal(0, sigma, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # 5) 블러 — 초점 흐림과 주행 중 모션 블러
    if rng.random() < 0.6:
        radius = rng.uniform(0.3, 1.8)
        applied["blur_radius"] = round(radius, 2)
        img = img.filter(ImageFilter.GaussianBlur(radius))

    # 6) 해상도 저하 — 원거리 촬영. 작게 줄였다가 다시 키워 정보 손실을 재현한다.
    if rng.random() < 0.5:
        scale = rng.uniform(0.25, 0.7)
        applied["downscale"] = round(scale, 2)
        small = img.resize((max(16, int(w * scale)), max(8, int(h * scale))),
                           Image.BILINEAR)
        img = small.resize((w, h), Image.BICUBIC)

    return img, applied


def to_ocr_input(img: Image.Image, width: int = 160, height: int = 32) -> Image.Image:
    """OCR 학습 입력 규격(32x160 회색조)으로 변환한다."""
    return img.convert("L").resize((width, height), Image.BICUBIC)


def main() -> None:
    parser = argparse.ArgumentParser(description="합성 번호판 생성기")
    parser.add_argument("--count", type=int, default=24, help="생성 장수")
    parser.add_argument("--out", type=str, default="data/synth", help="출력 디렉터리")
    parser.add_argument("--seed", type=int, default=42, help="난수 시드")
    args = parser.parse_args()

    out_dir = Path(args.out)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    _, hangul_ok = load_font(32)
    if not hangul_ok:
        print("[경고] 한글 폰트를 찾지 못했다. 용도문자/지역명이 깨져 보인다.")
        print("       Windows 는 맑은 고딕이, 리눅스는 fonts-nanum 패키지가 필요하다.")

    spec_names = list(SPECS.keys())
    records: list[dict] = []
    for i in range(args.count):
        spec_name = spec_names[i % len(spec_names)]
        text, palette_key = random_plate_text(spec_name, rng)
        clean = render_plate(text, spec_name, palette_key,
                             letter_spacing=rng.uniform(0.0, 0.12))
        noisy, applied = augment(clean, rng)
        final = to_ocr_input(noisy)

        rel = f"images/{i:05d}.png"
        final.save(out_dir / rel)
        records.append({"image": rel, "text": text.replace(" ", ""),
                        "raw_text": text, "spec": spec_name,
                        "palette": palette_key, "augment": applied})

    with (out_dir / "labels.jsonl").open("w", encoding="utf-8") as fp:
        for rec in records:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"생성 완료: {len(records)}장 → {out_dir.resolve()}")
    print(f"\n{'번호':3s} {'규격':18s} {'배경':11s} {'번호판':16s} 증강")
    print("-" * 92)
    for rec in records[:6]:
        aug = ", ".join(f"{k}={v}" for k, v in list(rec["augment"].items())[:3])
        print(f"{rec['image'][7:12]:3s} {rec['spec']:18s} {rec['palette']:11s} "
              f"{rec['raw_text']:16s} {aug}")

    from collections import Counter
    print("\n규격 분포:", dict(Counter(r["spec"] for r in records)))
    print("배경 분포:", dict(Counter(r["palette"] for r in records)))


if __name__ == "__main__":
    main()
