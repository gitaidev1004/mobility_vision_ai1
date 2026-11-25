/**
 * 숫자/날짜 포맷 유틸리티
 * - 화면 어디서나 동일한 표기를 보장하기 위해 포맷은 전부 이 파일에 모은다.
 * - 외부 라이브러리(dayjs, numeral 등)를 쓰지 않고 Intl API 와 순수 함수로 구현한다.
 */

/** 값이 숫자로 해석 가능한지 검사한다(null/undefined/'' 는 false). */
export function isNum(v) {
  return v !== null && v !== undefined && v !== '' && !Number.isNaN(Number(v))
}

/**
 * 천 단위 구분 기호를 넣은 정수/소수 문자열.
 * @param {number|string} v 값
 * @param {number} digits 소수점 자릿수(기본 0)
 * @returns {string} 예) 1234567 -> '1,234,567'
 */
export function formatNumber(v, digits = 0) {
  if (!isNum(v)) return '-'
  return Number(v).toLocaleString('ko-KR', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })
}

/**
 * 원화 금액. 기본은 '₩' 기호 대신 접미 '원' 을 쓴다(교재 지면 가독성).
 * @param {number|string} v 금액
 * @param {boolean} withSuffix true 면 '원' 을 붙인다.
 * @returns {string} 예) 184000 -> '184,000원'
 */
export function formatCurrency(v, withSuffix = true) {
  if (!isNum(v)) return '-'
  const n = Math.round(Number(v))
  return withSuffix ? `${n.toLocaleString('ko-KR')}원` : n.toLocaleString('ko-KR')
}

/**
 * 0~1 비율을 퍼센트 문자열로 변환한다.
 * @param {number} v 0~1 비율(이미 0~100 이면 asRatio=false)
 * @param {number} digits 소수점 자릿수(기본 1)
 * @param {boolean} asRatio v 가 0~1 비율인지 여부
 * @returns {string} 예) 0.962 -> '96.2%'
 */
export function formatPercent(v, digits = 1, asRatio = true) {
  if (!isNum(v)) return '-'
  const pct = asRatio ? Number(v) * 100 : Number(v)
  return `${pct.toFixed(digits)}%`
}

/** 두 자리 0 패딩. */
function pad2(n) {
  return String(n).padStart(2, '0')
}

/**
 * 어떤 입력이든 Date 로 바꾼다. 실패하면 null.
 * 백엔드는 ISO-8601(LocalDateTime, 타임존 없음)을 보내므로 그대로 파싱한다.
 */
export function toDate(v) {
  if (!v) return null
  if (v instanceof Date) return Number.isNaN(v.getTime()) ? null : v
  const d = new Date(v)
  return Number.isNaN(d.getTime()) ? null : d
}

/**
 * 'YYYY-MM-DD HH:mm' 형식의 일시 문자열.
 * @param {string|Date} v ISO 문자열 또는 Date
 * @returns {string} 예) '2026-10-24 14:05'
 */
export function formatDateTime(v) {
  const d = toDate(v)
  if (!d) return '-'
  return (
    `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}` +
    ` ${pad2(d.getHours())}:${pad2(d.getMinutes())}`
  )
}

/** 'YYYY-MM-DD' 형식의 날짜 문자열. */
export function formatDate(v) {
  const d = toDate(v)
  if (!d) return '-'
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`
}

/** 'HH:mm' 형식의 시각 문자열. */
export function formatTime(v) {
  const d = toDate(v)
  if (!d) return '-'
  return `${pad2(d.getHours())}:${pad2(d.getMinutes())}`
}

/**
 * 백엔드 쿼리 파라미터용 ISO LocalDateTime 문자열(초 포함, 타임존 없음).
 * @returns {string} 예) '2026-10-24T14:05:00'
 */
export function toLocalDateTimeParam(v) {
  const d = toDate(v)
  if (!d) return undefined
  return (
    `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}` +
    `T${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`
  )
}

/**
 * 밀리초 소요시간. 1000ms 미만은 ms, 그 이상은 초로 환산한다.
 * @param {number} ms 밀리초
 * @param {number} digits 소수점 자릿수(기본 1)
 * @returns {string} 예) 61.4 -> '61.4ms', 1830 -> '1.83s'
 */
export function formatDuration(ms, digits = 1) {
  if (!isNum(ms)) return '-'
  const n = Number(ms)
  if (n < 1000) return `${n.toFixed(digits)}ms`
  return `${(n / 1000).toFixed(2)}s`
}

/**
 * 주차 시간(분)을 'N시간 M분' 으로 표현한다.
 * @param {number} minutes 분
 * @returns {string} 예) 135 -> '2시간 15분', 42 -> '42분'
 */
export function formatMinutes(minutes) {
  if (!isNum(minutes)) return '-'
  const m = Math.max(0, Math.round(Number(minutes)))
  const h = Math.floor(m / 60)
  const rest = m % 60
  if (h === 0) return `${rest}분`
  if (rest === 0) return `${h}시간`
  return `${h}시간 ${rest}분`
}

/** 주행거리(km). */
export function formatKm(v) {
  if (!isNum(v)) return '-'
  return `${formatNumber(v)}km`
}

/** 'N일' 형식의 잔존수명(RUL) 표기. */
export function formatDays(v) {
  if (!isNum(v)) return '-'
  return `${formatNumber(v)}일`
}

/**
 * 상대 시간('3분 전'). 최근 이벤트 목록의 보조 표기로 사용한다.
 * @param {string|Date} v 기준 일시
 * @param {Date} now 현재 시각(테스트 주입용)
 */
export function formatRelative(v, now = new Date()) {
  const d = toDate(v)
  if (!d) return '-'
  const diffSec = Math.round((now.getTime() - d.getTime()) / 1000)
  if (diffSec < 0) return '곧'
  if (diffSec < 60) return `${diffSec}초 전`
  const min = Math.floor(diffSec / 60)
  if (min < 60) return `${min}분 전`
  const hour = Math.floor(min / 60)
  if (hour < 24) return `${hour}시간 전`
  const day = Math.floor(hour / 24)
  if (day < 30) return `${day}일 전`
  return formatDate(d)
}

/** 바이트 → 사람이 읽는 크기. 업로드 이미지 용량 표시에 사용한다. */
export function formatBytes(bytes) {
  if (!isNum(bytes)) return '-'
  const units = ['B', 'KB', 'MB', 'GB']
  let n = Number(bytes)
  let i = 0
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024
    i += 1
  }
  return `${n.toFixed(i === 0 ? 0 : 1)}${units[i]}`
}

/** 'hour' 숫자(0~23)를 '09시' 로 표기한다. */
export function formatHour(h) {
  if (!isNum(h)) return '-'
  return `${pad2(Number(h))}시`
}

/**
 * risk_level enum 을 한국어 라벨로 변환한다(계약서 2장 enum).
 */
export const RISK_LABEL = {
  LOW: '낮음',
  MEDIUM: '보통',
  HIGH: '높음',
  CRITICAL: '심각',
}

/**
 * plate_recognition.status enum 한국어 라벨.
 */
export const RECOGNITION_STATUS_LABEL = {
  SUCCESS: '성공',
  LOW_CONFIDENCE: '신뢰도 낮음',
  NO_PLATE: '번호판 미검출',
  FORMAT_INVALID: '형식 오류',
}

/** access_event.event_type 한국어 라벨. */
export const EVENT_TYPE_LABEL = {
  ENTRY: '입차',
  EXIT: '출차',
}

/** app_user.role 한국어 라벨. */
export const ROLE_LABEL = {
  ADMIN: '관리자',
  OPERATOR: '운영자',
  VIEWER: '조회자',
}

/** 결제 수단 한국어 라벨. */
export const PAYMENT_METHOD_LABEL = {
  CARD: '카드',
  CASH: '현금',
  MOBILE: '모바일',
  PASS: '정기권',
}

/** enum 값을 라벨 맵에서 찾되, 없으면 원본을 반환한다. */
export function labelOf(map, key) {
  if (!key) return '-'
  return map[key] ?? key
}
