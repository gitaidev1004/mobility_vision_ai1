/**
 * 인증 API (계약서 4장)
 * POST /api/v1/auth/login
 *   요청  : {"username","password"}
 *   응답  : {"accessToken","tokenType":"Bearer","expiresIn":3600,"username","role","displayName"}
 */
import http from './http'

/**
 * 로그인. 성공 시 토큰 응답 본문을 그대로 반환한다.
 * @param {{username:string, password:string}} payload
 */
export function login(payload) {
  return http.post('/auth/login', payload).then((r) => r.data)
}

/**
 * 현재 토큰의 사용자 정보 조회.
 * 백엔드가 /auth/me 를 제공하지 않을 수도 있으므로 404 는 null 로 흘린다.
 */
export function me() {
  return http
    .get('/auth/me')
    .then((r) => r.data)
    .catch((e) => {
      if (e?.response?.status === 404) return null
      throw e
    })
}

/**
 * 로그아웃. 서버 세션이 없는 JWT 방식이므로 실패해도 무시하고
 * 클라이언트 토큰만 지우면 충분하다(스토어가 처리).
 */
export function logout() {
  return http.post('/auth/logout').catch(() => null)
}
