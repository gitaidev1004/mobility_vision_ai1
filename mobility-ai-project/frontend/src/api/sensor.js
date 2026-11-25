/**
 * 센서 시계열 API (계약서 4장)
 * GET  /api/v1/sensors?vehicleId=&limit=100
 * POST /api/v1/sensors   (배열 적재)
 */
import http, { cleanParams } from './http'

/**
 * 차량별 센서 시계열 조회.
 * @param {{vehicleId:number, limit?:number}} params
 * @returns {Promise<Array>} sensor_reading 레코드 배열(measured_at 오름차순 가정)
 */
export function listSensors(params = {}) {
  const { vehicleId, limit = 100 } = params
  return http
    .get('/sensors', { params: cleanParams({ vehicleId, limit }) })
    .then((r) => r.data)
}

/**
 * 센서 데이터 배열 적재(시뮬레이터/게이트웨이가 호출하는 엔드포인트).
 * @param {Array<object>} readings sensor_reading 필드 배열
 */
export function ingestSensors(readings) {
  return http.post('/sensors', readings).then((r) => r.data)
}
