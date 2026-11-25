"""OCR(문자 인식) 패키지.

구성:
    - ``charset``       : 한국 번호판 문자셋 정의(숫자 / 용도문자 / 지역명)
    - ``crnn``          : PyTorch CRNN(CNN + BiLSTM + CTC) 모델 정의
    - ``ctc_decoder``   : CTC greedy / beam search 디코더(순수 numpy)
    - ``attention_ocr`` : CNN + Attention Seq2Seq OCR 모델 정의
    - ``paddle_adapter``: PaddleOCR 선택적 어댑터
    - ``recognizer``    : 인식기 팩토리 및 공통 인터페이스
    - ``postprocess``   : 번호판 정규식 검증, 혼동 보정, 다중 프레임 융합

``charset`` 과 ``ctc_decoder`` 는 numpy 외의 의존성이 없어 단위 테스트에서 바로 쓸 수 있다.
"""

__all__: list[str] = []
