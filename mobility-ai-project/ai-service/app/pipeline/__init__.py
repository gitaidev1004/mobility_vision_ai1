"""LPR 파이프라인 패키지.

구성:
    - ``quality``      : 블러(라플라시안 분산)/밝기/번호판 픽셀폭 품질평가
    - ``rectify``      : 기울기 보정, 투시 변환, CLAHE, 샤프닝
    - ``lpr_pipeline`` : 검출 → 보정 → OCR → 후처리 전체 오케스트레이션

``quality`` 는 numpy 만 필수이므로 단위 테스트에서 바로 실행된다.
"""

__all__: list[str] = []
