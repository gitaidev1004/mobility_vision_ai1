# 모빌리티 비전 AI — 프론트엔드

Vue 3.5 + Vite 6 + JavaScript 로 작성한 관리자 대시보드입니다.
차트/UI 라이브러리를 쓰지 않고 **순수 SVG 컴포넌트 + 직접 작성한 CSS** 로 구현했습니다.

## 실행

```bash
npm install
cp .env.example .env
npm run dev          # http://localhost:5173
```

`/api` 요청은 Vite 프록시가 백엔드(`http://localhost:8080`)로 전달합니다.
따라서 백엔드와 AI 서비스가 먼저 기동되어 있어야 합니다.

```bash
npm run build        # dist/ 로 정적 빌드
npm run preview      # 빌드 결과 미리보기
```

## 로그인

계약서 2장의 시드 계정을 사용합니다. 로그인 화면의 버튼을 누르면 입력란이 자동으로 채워집니다.

| 계정 | 비밀번호 | 역할 |
|---|---|---|
| `admin` | `admin123!` | ADMIN (전체) |
| `operator` | `operator123!` | OPERATOR (운영·정산) |
| `viewer` | `viewer123!` | VIEWER (조회 전용) |

## 라우트

| 경로 | 화면 | 주요 기능 |
|---|---|---|
| `/login` | 로그인 | JWT 발급, 시드 계정 바로 입력 |
| `/` | 대시보드 | 통계 카드 8개, 시간대별 교통량, 위험도 도넛, 최근 이벤트 |
| `/recognition` | 번호판 인식 | 이미지 업로드, 검출 박스 오버레이, 결과 패널, 인식 이력 |
| `/vehicles` | 차량 관리 | 검색·페이지, 등록/수정 모달, 삭제 확인 |
| `/events` | 출입 이벤트 | 구분·기간 필터, 게이트별 집계 |
| `/parking` | 주차 · 정산 | 점유 현황, 시간대별 입출차, 세션 목록, 정산 모달 |
| `/maintenance` | 정비 이력 | 차량별 조회, 항목별 비용, 등록 모달 |
| `/pdm` | 예지보전 | 건강점수 게이지, 센서 추이, 부품별 점수, 정비 권고 |
| `/benchmark` | 모델 성능 | 비교표, 지연시간·정확도·크기 막대그래프 |

토큰이 없으면 라우터 가드가 `/login` 으로 보냅니다(`src/router/index.js`).

## 구조

```
src/
├── api/        axios 인스턴스 + 도메인별 API 함수 (계약서 4장과 1:1)
├── stores/     pinia 스토어 (auth, dashboard, vehicle, recognition, pdm)
├── components/
│   ├── layout/ 사이드바 · 상단바 · 셸
│   ├── common/ 카드 · 표 · 배지 · 모달 · 토스트 · 페이지네이션
│   └── charts/ BarChart · LineChart · DonutChart · GaugeChart (전부 순수 SVG)
├── views/      라우트별 화면
├── utils/      format.js(숫자·날짜 포맷), icons.js(인라인 SVG), toast.js
└── assets/styles/  tokens.css → base.css → components.css
```

- `api/http.js`: baseURL `/api/v1`, 요청 인터셉터로 `Bearer` 주입,
  401 이면 토큰을 비우고 커스텀 이벤트로 전역에 알립니다(`App.vue` 가 `/login` 으로 이동).
- 색상·간격·타이포그래피는 모두 `tokens.css` 의 CSS 변수를 참조합니다.
- 1280px 이하에서 사이드바가 아이콘만 남기고 72px 로 축소됩니다.

## 교재 수록용 프리렌더

`prerender/` 에 빌드 없이 브라우저에서 바로 열리는 정적 HTML 5개가 있습니다.
Vue 컴포넌트와 동일한 마크업·CSS 를 사용하며, 외부 네트워크 리소스를 전혀 참조하지 않습니다.

```
prerender/_shared.css
prerender/{dashboard,recognition,parking,pdm,benchmark}.html
```

SVG 차트는 좌표가 미리 계산된 완성형이라 자바스크립트 없이도 그대로 보입니다.

## 백엔드 계약

응답 필드명은 `CONTRACT.md` 4장과 정확히 일치합니다(camelCase).
페이지 응답은 `{content, totalElements, totalPages, page, size}` 형태를 가정합니다.
