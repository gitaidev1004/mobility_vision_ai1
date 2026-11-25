"""FastAPI 라우터 패키지.

구성:
    - ``health`` : ``GET /health`` (prefix 없음 — 계약서 1장)
    - ``lpr``    : ``POST /api/v1/lpr/recognize``, ``/recognize-frames``
    - ``pdm``    : ``POST /api/v1/pdm/predict``
    - ``models`` : ``GET /api/v1/models/info``, ``POST /api/v1/models/benchmark``

라우터는 HTTP 관심사만 담당한다 — 멀티파트 파싱, 상태 코드, 응답 직렬화.
추론 로직은 ``app.pipeline`` 과 ``app.pdm`` 에 있다(관심사 분리).
"""

__all__: list[str] = []
