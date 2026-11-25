# 프론트엔드 파일 목록 (62개)

Vue 3.5 + Vite 6 + JavaScript. 차트/UI 라이브러리 없음(순수 SVG + 직접 작성 CSS).
런타임 의존성은 `vue`, `vue-router`, `pinia`, `axios` 네 개뿐입니다.

## 루트 설정

| 파일 | 설명 |
|---|---|
| `package.json` | vue 3.5.13 / vite 6.0.7 / @vitejs/plugin-vue 5.2.1 / vue-router 4.5.0 / pinia 2.3.1 / axios 1.12.2 |
| `vite.config.js` | server.port 5173 고정, `/api` → `http://localhost:8080` 프록시, `@` → `src` 별칭 |
| `index.html` | Vue 마운트 지점(`#app`), 한국어 lang, favicon 연결 |
| `.env.example` | `VITE_API_BASE_URL`, 백엔드 주소, 주차 수용 대수, 대시보드 자동 갱신 주기 |
| `README.md` | 실행 방법, 시드 계정, 라우트 표, 디렉터리 구조 설명 |
| `MANIFEST.md` | 이 파일 |
| `public/favicon.svg` | 차량 + 녹색 번호판 검출 박스 로고(인라인 SVG) |

## 진입점 · 라우팅

| 파일 | 설명 |
|---|---|
| `src/main.js` | pinia·router 설치, 전역 CSS 를 토큰→기본→컴포넌트 순으로 로드 |
| `src/App.vue` | 로그인은 레이아웃 없이, 그 외는 `AppLayout` 안에서 렌더. 401 이벤트 수신 후 강제 로그아웃 |
| `src/router/index.js` | 9개 라우트 정의, `meta.requiresAuth` 기반 인증 가드, 동적 import 코드 분할, 탭 제목 갱신 |

## API 계층 (계약서 4장과 1:1)

| 파일 | 설명 |
|---|---|
| `src/api/http.js` | axios 인스턴스(baseURL `/api/v1`), Bearer 주입, 401 → 세션 삭제 + 커스텀 이벤트, 오류 메시지 추출, 빈 파라미터 제거 |
| `src/api/auth.js` | `POST /auth/login`, `/auth/me`, `/auth/logout` |
| `src/api/vehicle.js` | 차량 목록/단건/번호판 조회/등록/수정/삭제 |
| `src/api/recognition.js` | `POST /recognitions`(multipart, 업로드 진행률), 인식 이력, 실패 목록 |
| `src/api/event.js` | `GET /events` 구분·기간·페이지 필터 |
| `src/api/parking.js` | 주차 세션 목록, 세션 정산, 주차장 현황 |
| `src/api/maintenance.js` | 정비 이력 조회·등록 |
| `src/api/sensor.js` | 센서 시계열 조회, 배열 적재 |
| `src/api/pdm.js` | 예측 실행, 최신 예측(404→null), 위험 차량 목록 |
| `src/api/dashboard.js` | `GET /dashboard/summary` |
| `src/api/benchmark.js` | 벤치마크 목록, 재측정 실행(타임아웃 120초) |

## Pinia 스토어

| 파일 | 설명 |
|---|---|
| `src/stores/auth.js` | 토큰·사용자 정보를 localStorage 와 동기화, 역할별 권한 계산(`canWrite`/`isAdmin`) |
| `src/stores/dashboard.js` | 요약 응답 보관 + 통계 카드 8개, 0~23시 교통량 보정, 위험도 도넛 조각 변환 |
| `src/stores/vehicle.js` | 검색어·페이지 상태, 목록/등록/수정/삭제(삭제 시 빈 페이지 보정) |
| `src/stores/recognition.js` | 파일 선택·미리보기 URL 관리, 검출 박스를 퍼센트 좌표로 정규화, 인식 실행 및 이력 |
| `src/stores/pdm.js` | 차량 선택 → 예측+센서 병렬 로드, 부품/권고 정렬, 센서 시리즈 5종 구성 |

## 유틸리티

| 파일 | 설명 |
|---|---|
| `src/utils/format.js` | 원화·퍼센트·`YYYY-MM-DD HH:mm`·소요시간(ms)·주차시간·상대시간 포맷 + enum 한국어 라벨 맵 |
| `src/utils/icons.js` | 24×24 인라인 SVG path 28종(아이콘 폰트 미사용) |
| `src/utils/toast.js` | 전역 reactive 토스트 큐(성공/경고/오류 단축 함수) |

## 전역 스타일

| 파일 | 설명 |
|---|---|
| `src/assets/styles/tokens.css` | 색상·간격·반경·그림자·타이포 CSS 변수(계약서 5장 색상 규약 반영) |
| `src/assets/styles/base.css` | reset, 한글 폰트 스택, 포커스 링, `.sr-only`, 스크롤바 |
| `src/assets/styles/components.css` | 레이아웃 셸·카드·버튼·폼·표·배지·모달·토스트·차트 공통 + 1280px 사이드바 축소 미디어쿼리 |

## 레이아웃 컴포넌트

| 파일 | 설명 |
|---|---|
| `src/components/layout/AppLayout.vue` | 사이드바 + {상단바 + 본문} 셸, 진입 시 대시보드 요약 1회 로드 |
| `src/components/layout/AppSidebar.vue` | 운영/자산·분석 2그룹 메뉴, 현재 라우트 강조, 미정산 배지 |
| `src/components/layout/AppTopbar.vue` | 라우트 제목, 1초 주기 시계, 로그인 사용자, 로그아웃 버튼 |

## 공통 컴포넌트

| 파일 | 설명 |
|---|---|
| `src/components/common/AppIcon.vue` | `icons.js` path 를 stroke 방식으로 그리는 인라인 SVG 아이콘 |
| `src/components/common/PageHeader.vue` | 화면 제목 + 설명 + 우측 액션 슬롯 |
| `src/components/common/StatCard.vue` | 라벨·큰 숫자·단위·증감 표기, number/currency/percent/duration/text 포맷 |
| `src/components/common/StatusBadge.vue` | 계약서 enum 8종(recognition/risk/event/component/allowed/paid/active/parked)을 한국어 배지로 변환 |
| `src/components/common/DataTable.vue` | 열 선언형 표, 열별 커스텀 슬롯, 로딩 스켈레톤, 빈 상태, `<caption>`/`scope` 필수 출력 |
| `src/components/common/PaginationBar.vue` | 0-base 페이지, 현재 위치 주변 5개 노출, 'N–M / 총 K건' 안내 |
| `src/components/common/EmptyState.vue` | 아이콘 + 제목 + 설명 + 액션 슬롯 |
| `src/components/common/ConfirmDialog.vue` | 삭제·정산 확인 다이얼로그, ESC 닫힘, 열릴 때 확인 버튼 포커스 |
| `src/components/common/ToastHost.vue` | 우측 하단 토스트 스택, `aria-live="polite"` |

## 차트 컴포넌트 (전부 순수 SVG, 외부 라이브러리 0)

| 파일 | 설명 |
|---|---|
| `src/components/charts/BarChart.vue` | 세로/가로 막대, 다중 시리즈 그룹, 눈금 자동 올림(1·1.5·2·2.5·3·4·5·6·8·10 단계), 값 라벨, `labelWidth` 조절 |
| `src/components/charts/LineChart.vue` | 다중 시리즈 꺾은선, 영역 채움, 데이터 점, zero-based/자동 y범위 전환 |
| `src/components/charts/DonutChart.vue` | arc 패스 직접 계산(조각 간격 제어), 중앙 합계, 값+비율 범례 |
| `src/components/charts/GaugeChart.vue` | 240도 스윕 계기판, 배경·값 호, 눈금, 바늘(라벨과 겹치지 않도록 반지름 분리) |

## 화면 (views)

| 파일 | 설명 |
|---|---|
| `src/views/LoginView.vue` | 좌측 브랜드 패널 + 우측 폼, 시드 계정 3개 원클릭 입력, `role="alert"` 오류 |
| `src/views/DashboardView.vue` | 통계 카드 8개 + 시간대별 교통량(꺾은선·막대) + 위험도 도넛 + 최근 이벤트 표, 자동 갱신 |
| `src/views/RecognitionView.vue` | 드래그&드롭 업로드, 검출 박스 퍼센트 오버레이, 결과 패널(소요시간·품질·문자별 신뢰도), 검출 상세표, 인식 이력 |
| `src/views/VehiclesView.vue` | 검색·페이지 목록, 등록/수정 2열 모달(번호판 정규식 검증), 삭제 확인 |
| `src/views/EventsView.vue` | 구분·기간 필터, 오늘/최근 7일 단축 버튼, 게이트별 입출차 집계 |
| `src/views/ParkingView.vue` | 현황 카드 4개, 점유율 도넛, 시간대별 입출차, 세션 목록, 정산 모달(할인·최종 금액) |
| `src/views/MaintenanceView.vue` | 비용 요약 카드, 항목별 비용 가로 막대, 차량 필터 목록, 등록 모달 |
| `src/views/PdmView.vue` | 차량 선택 → 건강점수 게이지 + 부품별 점수/사유 + 센서 추이 4종 + 타이어 공기압 + 정비 권고 + 위험 차량 |
| `src/views/BenchmarkView.vue` | 최고 성능 요약, 비교표(최고값 ★ 강조), 지연시간·정확도·모델 크기 막대그래프 |

## 교재 수록용 프리렌더 (`prerender/`)

빌드·네트워크 없이 브라우저에서 바로 열립니다. Vue 컴포넌트와 동일한 마크업·CSS 를 쓰고,
SVG 차트는 좌표가 미리 계산된 완성형이라 자바스크립트 실행이 필요 없습니다.
1440×900 기준 고정 폭 레이아웃, 라이트 테마, 외부 리소스 참조 0건(검증 완료).

| 파일 | 설명 |
|---|---|
| `prerender/_shared.css` | tokens + base + components + 각 `.vue` scoped 스타일을 de-scope 해 합친 단일 CSS |
| `prerender/dashboard.html` | 통계 카드 8개, 24시간 교통량 꺾은선·막대, 위험도 도넛, 최근 이벤트 8건 |
| `prerender/recognition.html` | CSS/SVG 로 그린 차량 플레이스홀더 + 녹색 검출 박스 `plate 0.98` + `12가3456 (97.3%)` 결과, 검출 상세, 이력 10건 |
| `prerender/parking.html` | 점유 현황 4카드, 점유율 도넛(31%), 시간대별 입출차, 세션 10건(정산/미정산 혼합) |
| `prerender/pdm.html` | 건강점수 게이지(58.6), 부품 7종 점수, 센서 추이 4종(30일), 타이어 공기압, 권고 4건, 위험 차량 5대 |
| `prerender/benchmark.html` | 모델 6종 비교표(최고값 ★), 지연시간 p50/p95, 정확도·모델 크기 가로 막대 |

## 검증 스크립트 (`/home/claude/work/scripts/`)

| 파일 | 설명 |
|---|---|
| `check_js.sh` | 모든 `.js` 를 `.mjs` 로 복사해 `node --check` 로 문법 검증 후 복사본 삭제 (22/22 통과) |
| `check_vue.py` | `.vue` 블록 구조·태그 짝·중괄호 균형·보간 짝·LF 검사 (26/26 통과) |
| `gen_prerender.py` | 차트 컴포넌트의 좌표 계산 로직을 파이썬으로 이식해 프리렌더 HTML/CSS 생성 |

### 검증 결과

- `node --check`: JS 22개 전부 통과
- `.vue` 구조 검사: 26개 전부 통과
- Playwright 1440×900 렌더: 5개 페이지 모두 **가로 오버플로 없음, 콘솔 오류 0건, 외부 요청 0건**
  (높이 dashboard 1493 / recognition 1643 / parking 1400 / pdm 2497 / benchmark 1809)
- 1100px 뷰포트에서 사이드바 72px 축소 동작 확인
- 스크린샷 육안 검증 완료(레이아웃 깨짐·겹침·잘린 텍스트·빈 영역 없음)
