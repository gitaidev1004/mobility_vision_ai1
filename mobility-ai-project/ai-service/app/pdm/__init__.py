"""예지보전(PdM, Predictive Maintenance) 패키지.

구성:
    - ``features`` : 결측/이상값 처리, 이동평균, rolling 통계, 윈도우 생성
    - ``anomaly``  : IsolationForest / OneClassSVM / AutoEncoder + 통계 임계
    - ``failure``  : RandomForest 류 고장 분류(sklearn 기본, 미설치 대비 규칙 기반)
    - ``rul``      : LSTM 기반 잔여수명(RUL) + 회귀 폴백
    - ``service``  : 위 모듈을 묶어 ``/pdm/predict`` 응답을 생성

전체 모듈이 **numpy 만으로 동작**한다. sklearn/torch 가 있으면 더 정교한 모델을
쓰고, 없으면 도메인 규칙과 통계 기법으로 대체한다. 어느 경로에서도 같은 스키마를
반환하므로 백엔드/프론트엔드는 차이를 알 필요가 없다.
"""

__all__: list[str] = []
