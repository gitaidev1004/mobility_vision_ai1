"""번호판 검출(detection) 패키지.

구성:
    - ``base``              : ``PlateDetector`` 추상 클래스와 ``Detection`` 데이터 구조
    - ``preprocess``        : letterbox / resize / NMS 등 전처리·후처리 유틸
    - ``yolo_detector``     : ultralytics YOLO + ONNXRuntime 양쪽 경로 지원 래퍼
    - ``mobilenet_detector``: torchvision SSDLite320-MobileNetV3 기반 경량 검출기

모든 검출기는 가중치가 없으면 ``app.core.fallback`` 의 OpenCV 검출로 자동 대체된다.
"""

__all__: list[str] = []
