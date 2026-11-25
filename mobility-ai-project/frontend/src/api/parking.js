/**
 * 주차 API (계약서 4장)
 * GET  /api/v1/parking/sessions?active=true&page=&size=
 * POST /api/v1/parking/sessions/{id}/pay   body {"paymentMethod":"CARD","discountAmount":0}
 * GET  /api/v1/parking/status
 *      → {capacity, occupied, available, occupancyRate, unpaidCount, byHour:[{hour,entries,exits}]}
 */
import http, { cleanParams } from './http'

/**
 * 주차 세션 목록.
 * @param {{active?:boolean, page?:number, size?:number}} params
 *        active=true 면 출차하지 않은(주차 중) 세션만 조회한다.
 */
export function listSessions(params = {}) {
  const { active, page = 0, size = 20 } = params
  return http
    .get('/parking/sessions', { params: cleanParams({ active, page, size }) })
    .then((r) => r.data)
}

/**
 * 주차 요금 정산.
 * @param {number} id 주차 세션 ID
 * @param {{paymentMethod:string, discountAmount:number}} payload
 *        paymentMethod ∈ CARD|CASH|MOBILE|PASS
 */
export function paySession(id, payload) {
  const body = {
    paymentMethod: payload?.paymentMethod ?? 'CARD',
    discountAmount: Number(payload?.discountAmount ?? 0),
  }
  return http.post(`/parking/sessions/${id}/pay`, body).then((r) => r.data)
}

/** 주차장 현황(수용/점유/시간대별 입출차). */
export function getStatus() {
  return http.get('/parking/status').then((r) => r.data)
}
