/**
 * 정비 이력 API (계약서 4장)
 * GET  /api/v1/maintenance?vehicleId=&page=&size=
 * POST /api/v1/maintenance
 */
import http, { cleanParams } from './http'

/**
 * 정비 이력 조회.
 * @param {{vehicleId?:number, page?:number, size?:number}} params
 */
export function listMaintenance(params = {}) {
  const { vehicleId, page = 0, size = 20 } = params
  return http
    .get('/maintenance', { params: cleanParams({ vehicleId, page, size }) })
    .then((r) => r.data)
}

/**
 * 정비 등록.
 * @param {object} payload vehicleId, serviceDate, odometerKm, item,
 *                         partName, cost, shopName, memo
 */
export function createMaintenance(payload) {
  return http.post('/maintenance', payload).then((r) => r.data)
}
