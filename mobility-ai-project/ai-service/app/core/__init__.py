"""공통 코어 유틸리티 패키지.

- ``logging``  : 구조화 로깅 + 요청 ID 컨텍스트
- ``registry`` : 모델 레지스트리(이름 → 로더). 지연 로딩 + 싱글턴 캐시
- ``fallback`` : 가중치가 없을 때 쓰는 결정적 OpenCV 추론

의도적으로 서브모듈을 재export 하지 않는다. ``app.core`` 를 import 하는 것만으로
OpenCV/torch 가 끌려오면 "의존성 없이도 import 가능" 원칙이 깨지기 때문이다.
"""

__all__: list[str] = []
