/**
 * axios 공용 인스턴스 (계약서 5장)
 * - baseURL: '/api/v1' (Vite 프록시가 http://localhost:8080 으로 전달)
 * - 요청 인터셉터: localStorage 의 액세스 토큰을 Authorization 헤더에 주입
 * - 응답 인터셉터: 401 이면 토큰을 비우고 /login 으로 보낸다.
 *   (스토어를 직접 import 하면 순환 참조가 생기므로 커스텀 이벤트로 알린다.)
 */
import axios from 'axios'

/** 액세스 토큰 localStorage 키 */
export const TOKEN_KEY = 'mobility.accessToken'
/** 로그인 사용자 정보 localStorage 키 */
export const USER_KEY = 'mobility.user'

/** 인증 만료를 전역에 알리는 커스텀 이벤트 이름 */
export const UNAUTHORIZED_EVENT = 'mobility:unauthorized'

/** 저장된 토큰을 읽는다. 브라우저가 아닌 환경(SSR/테스트)에서는 null. */
export function readToken() {
  try {
    return localStorage.getItem(TOKEN_KEY)
  } catch {
    return null
  }
}

/** 토큰을 저장한다. null 을 넘기면 삭제한다. */
export function writeToken(token) {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token)
    else localStorage.removeItem(TOKEN_KEY)
  } catch {
    /* 프라이빗 모드 등에서 저장이 막힐 수 있으므로 무시한다. */
  }
}

/** 저장된 사용자 정보를 읽는다. */
export function readUser() {
  try {
    const raw = localStorage.getItem(USER_KEY)
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

/** 사용자 정보를 저장한다. null 이면 삭제. */
export function writeUser(user) {
  try {
    if (user) localStorage.setItem(USER_KEY, JSON.stringify(user))
    else localStorage.removeItem(USER_KEY)
  } catch {
    /* 무시 */
  }
}

/** 공용 axios 인스턴스 */
const http = axios.create({
  baseURL: import.meta.env?.VITE_API_BASE_URL || '/api/v1',
  timeout: 30000,
  headers: {
    Accept: 'application/json',
  },
})

// ---------------- 요청 인터셉터 ----------------
http.interceptors.request.use((config) => {
  const token = readToken()
  if (token) {
    config.headers = config.headers || {}
    config.headers.Authorization = `Bearer ${token}`
  }
  // FormData 전송 시에는 브라우저가 boundary 를 포함한 Content-Type 을 직접 설정해야 하므로
  // axios 가 임의로 지정한 헤더를 제거한다.
  if (typeof FormData !== 'undefined' && config.data instanceof FormData) {
    delete config.headers['Content-Type']
  }
  return config
})

// ---------------- 응답 인터셉터 ----------------
http.interceptors.response.use(
  (res) => res,
  (error) => {
    const status = error?.response?.status

    // 401: 토큰 만료/위조 → 로컬 세션을 지우고 로그인 화면으로 유도한다.
    // 단, 로그인 요청 자체의 401(=비밀번호 오류)은 화면에서 처리해야 하므로 제외한다.
    const url = error?.config?.url || ''
    const isLoginCall = url.includes('/auth/login')

    if (status === 401 && !isLoginCall) {
      writeToken(null)
      writeUser(null)
      if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent(UNAUTHORIZED_EVENT))
      }
    }

    // GlobalExceptionHandler 가 보내는 공통 오류 스키마
    // {"timestamp","status","error","message","path"} 에서 message 를 꺼내 붙인다.
    error.friendlyMessage = extractMessage(error)
    return Promise.reject(error)
  },
)

/**
 * 백엔드 공통 오류 응답에서 사용자에게 보여줄 메시지를 뽑아낸다.
 * @param {unknown} error axios 오류 객체
 * @returns {string} 한국어 메시지
 */
export function extractMessage(error) {
  const data = error?.response?.data
  if (data) {
    if (typeof data === 'string') return data
    if (data.message) return data.message
    if (data.error) return data.error
  }
  if (error?.code === 'ECONNABORTED') return '요청 시간이 초과되었습니다. 잠시 후 다시 시도하세요.'
  if (!error?.response) return '서버에 연결할 수 없습니다. 백엔드(8080) 상태를 확인하세요.'
  return error?.message || '알 수 없는 오류가 발생했습니다.'
}

/**
 * undefined/null/'' 인 쿼리 파라미터를 제거한다.
 * 백엔드가 빈 문자열을 enum 으로 변환하려다 400 을 던지는 것을 막는다.
 * @param {Record<string, unknown>} params
 */
export function cleanParams(params = {}) {
  const out = {}
  Object.entries(params).forEach(([k, v]) => {
    if (v === undefined || v === null || v === '') return
    out[k] = v
  })
  return out
}

export default http
