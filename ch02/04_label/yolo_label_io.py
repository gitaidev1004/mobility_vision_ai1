# -*- coding: utf-8 -*-
"""YOLO 라벨 포맷 읽기/쓰기/좌표 변환 모듈.

YOLO 라벨 한 줄의 구조:
    <class_id> <cx> <cy> <w> <h>
  - class_id: 0 부터 시작하는 정수
  - cx, cy  : 박스 중심의 이미지 대비 비율 (0.0 ~ 1.0)
  - w,  h   : 박스 폭/높이의 이미지 대비 비율 (0.0 ~ 1.0)

정규화 좌표를 쓰는 이유는 이미지 크기가 달라져도 라벨을 그대로 쓸 수 있기
때문이다. 반대로 사람이 눈으로 검증하기는 어려우므로 픽셀 좌표로 바꿔 보는
변환 함수를 함께 둔다.

표준 라이브러리만 사용하므로 다른 스크립트에서 import 해 재사용할 수 있다.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass


@dataclass(frozen=True)
class YoloBox:
    """YOLO 라벨 한 줄을 나타내는 값 객체."""

    cls: int
    cx: float
    cy: float
    w: float
    h: float

    def to_pixel(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        """정규화 좌표를 (x1, y1, x2, y2) 픽셀 좌표로 바꾼다."""
        x1 = (self.cx - self.w / 2.0) * img_w
        y1 = (self.cy - self.h / 2.0) * img_h
        x2 = (self.cx + self.w / 2.0) * img_w
        y2 = (self.cy + self.h / 2.0) * img_h
        return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))

    def pixel_size(self, img_w: int, img_h: int) -> tuple[int, int]:
        """박스의 픽셀 폭/높이를 돌려준다."""
        return int(round(self.w * img_w)), int(round(self.h * img_h))

    @property
    def aspect_ratio(self) -> float:
        """정규화 좌표 기준 w/h. 이미지 종횡비가 섞여 있으므로 비교용으로만 쓴다."""
        return self.w / self.h if self.h > 0 else 0.0

    def pixel_aspect_ratio(self, img_w: int, img_h: int) -> float:
        """실제 픽셀 기준 종횡비. 한국 번호판은 대략 2.0~5.3 범위다.

        라벨 검수에서 이 값이 범위를 크게 벗어나면 박스를 잘못 그린 것이다.
        """
        pw, ph = self.pixel_size(img_w, img_h)
        return pw / ph if ph > 0 else 0.0

    def as_line(self) -> str:
        """라벨 파일에 저장할 한 줄 문자열로 만든다(소수점 6자리)."""
        return f"{self.cls} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


def from_pixel(cls: int, x1: float, y1: float, x2: float, y2: float,
               img_w: int, img_h: int) -> YoloBox:
    """픽셀 박스를 YOLO 정규화 박스로 바꾼다.

    좌표가 뒤집혀 들어오는 실수가 흔하므로 먼저 정렬한다.
    """
    x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
    y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
    # 이미지 밖으로 나간 좌표를 경계로 잘라낸다
    x1 = max(0.0, min(float(img_w), x1))
    x2 = max(0.0, min(float(img_w), x2))
    y1 = max(0.0, min(float(img_h), y1))
    y2 = max(0.0, min(float(img_h), y2))
    return YoloBox(cls,
                   (x1 + x2) / 2.0 / img_w,
                   (y1 + y2) / 2.0 / img_h,
                   (x2 - x1) / img_w,
                   (y2 - y1) / img_h)


def read_labels(path: str | pathlib.Path) -> list[YoloBox]:
    """라벨 파일을 읽는다. 파일이 없으면 빈 목록(배경 이미지)을 돌려준다."""
    p = pathlib.Path(path)
    if not p.exists():
        return []
    out: list[YoloBox] = []
    for ln, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if not s:
            continue                      # 빈 줄은 건너뛴다
        parts = s.split()
        if len(parts) != 5:
            raise ValueError(f"{p}:{ln} 토큰 수가 5개가 아니다: {s!r}")
        try:
            out.append(YoloBox(int(parts[0]), float(parts[1]), float(parts[2]),
                               float(parts[3]), float(parts[4])))
        except ValueError as exc:
            raise ValueError(f"{p}:{ln} 숫자 변환 실패: {s!r}") from exc
    return out


def write_labels(path: str | pathlib.Path, boxes: list[YoloBox]) -> None:
    """라벨 파일을 쓴다. 박스가 없으면 빈 파일을 만든다(배경 이미지 표시)."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(b.as_line() + "\n" for b in boxes), encoding="utf-8")


def pascal_voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h, cls=0) -> YoloBox:
    """Pascal VOC XML 의 픽셀 좌표를 YOLO 로 바꾼다(변환 도구 이식 시 사용)."""
    return from_pixel(cls, xmin, ymin, xmax, ymax, img_w, img_h)


def coco_to_yolo(x, y, w, h, img_w, img_h, cls=0) -> YoloBox:
    """COCO 의 (x, y, w, h) 좌상단 기준 좌표를 YOLO 로 바꾼다."""
    return from_pixel(cls, x, y, x + w, y + h, img_w, img_h)


def main() -> None:
    img_w, img_h = 1920, 1080

    # 라벨링 도구가 내놓은 픽셀 박스 세 건을 YOLO 로 변환한다
    pixel_boxes = [
        (0, 412, 288, 596, 342),     # 12가3456 — 정상
        (0, 1300, 700, 1420, 736),   # 87로2345 — 소형
        (0, 60, 900, 410, 1000),     # 서울 34나5678 — 대형
    ]
    print("=== 픽셀 → YOLO 정규화 변환 ===")
    boxes: list[YoloBox] = []
    for cls, x1, y1, x2, y2 in pixel_boxes:
        b = from_pixel(cls, x1, y1, x2, y2, img_w, img_h)
        boxes.append(b)
        pw, ph = b.pixel_size(img_w, img_h)
        print(f"  픽셀({x1},{y1},{x2},{y2}) -> {b.as_line()}")
        print(f"     복원 크기 {pw}x{ph}px  "
              f"픽셀 종횡비 {b.pixel_aspect_ratio(img_w, img_h):.2f}")

    # 라벨 파일에 쓰고 다시 읽어 왕복(round-trip) 일치를 확인한다
    out = pathlib.Path("data/demo_labels/sample001.txt")
    write_labels(out, boxes)
    print(f"\n=== 저장 결과 {out} ===")
    print(out.read_text(encoding="utf-8").rstrip())

    loaded = read_labels(out)
    print("\n=== 왕복 검증 (저장 → 로드 → 픽셀 복원) ===")
    ok = True
    for orig, back in zip(pixel_boxes, loaded):
        px = back.to_pixel(img_w, img_h)
        same = px == tuple(orig[1:])
        ok = ok and same
        print(f"  원본 {tuple(orig[1:])} -> 복원 {px}  일치={same}")
    print("  전체 일치:", ok)

    # 다른 포맷에서의 변환 예시
    print("\n=== 타 포맷 변환 ===")
    print("  VOC (412,288,596,342) ->",
          pascal_voc_to_yolo(412, 288, 596, 342, img_w, img_h).as_line())
    print("  COCO (412,288,184,54) ->",
          coco_to_yolo(412, 288, 184, 54, img_w, img_h).as_line())

    # 배경 이미지(번호판 없음)는 빈 라벨 파일로 표현한다
    bg = pathlib.Path("data/demo_labels/background001.txt")
    write_labels(bg, [])
    print("\n배경 이미지 라벨:", repr(bg.read_text(encoding="utf-8")),
          "-> 박스 수", len(read_labels(bg)))


if __name__ == "__main__":
    main()
