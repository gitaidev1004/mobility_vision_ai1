/**
 * 출입 이벤트 API (계약서 4장)
 * GET /api/v1/events?type=&from=&to=&page=&size=
 *   type ∈ ENTRY|EXIT
 */
import http, { cleanParams } from './http'

/**
 * 출입 이벤트 페이지 조회.
 * @param {{type?:'ENTRY'|'EXIT', from?:string, to?:string, page?:number, size?:number}} params
 * @returns {Promise<{content:Array, totalElements:number, totalPages:number, page:number, size:number}>}
 */
export function listEvents(params = {}) {
  const { type, from, to, page = 0, size = 20 } = params
  return http
    .get('/events', { params: cleanParams({ type, from, to, page, size }) })
    .then((r) => r.data)
}
