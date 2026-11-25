"""학습 / 내보내기 / 벤치마크 CLI 패키지.

구성:
    - ``synth_plate``    : 합성 한국 번호판 이미지 생성기(PIL, 증강 포함)
    - ``train_detector`` : YOLO 검출기 학습 스크립트
    - ``train_crnn``     : CRNN + CTC 학습 루프(체크포인트/EarlyStopping)
    - ``export_onnx``    : ONNX 내보내기
    - ``quantize``       : INT8 동적 양자화
    - ``benchmark``      : CPU 지연/메모리/정확도 벤치마크

모든 스크립트는 ``python -m app.train.<모듈>`` 로 실행된다.
torch/ultralytics 가 없으면 **친절한 안내 메시지와 함께 종료**하며, import 단계에서
터지지 않는다(``synth_plate`` 와 ``benchmark`` 는 torch 없이도 완전히 동작한다).
"""

__all__: list[str] = []
