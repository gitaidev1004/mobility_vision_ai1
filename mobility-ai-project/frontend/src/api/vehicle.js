/**
 * 차량 API (계약서 4장)
 * GET    /api/v1/vehicles?q=&page=0&size=20   → {content,totalElements,totalPages,page,size}
 * POST   /api/v1/vehicles
 * GET    /api/v1/vehicles/{id}
 * PUT    /api/v1/vehicles/{id}
 * DELETE /api/v1/vehicles/{id}
 * GET    /api/v1/vehicles/by-plate/{plate}
 */
import http, { cleanParams } from './http'

/**
 * 차량 페이지 목록.
 * @param {{q?:string, page?:number, size?:number}} params
 */
export function listVehicles(params = {}) {
  const { q = '', page = 0, size = 20 } = params
  return http
    .get('/vehicles', { params: cleanParams({ q, page, size }) })
    .then((r) => r.data)
}

/** 단건 조회. */
export function getVehicle(id) {
  return http.get(`/vehicles/${id}`).then((r) => r.data)
}

/**
 * 번호판으로 조회. 한글이 포함되므로 반드시 encodeURIComponent 를 거친다.
 * @param {string} plate 예) '12가3456'
 */
export function getVehicleByPlate(plate) {
  return http.get(`/vehicles/by-plate/${encodeURIComponent(plate)}`).then((r) => r.data)
}

/**
 * 차량 등록.
 * @param {object} payload plateNumber, ownerName, phone, vehicleType, modelName,
 *                         modelYear, vin, passType, passExpiresAt, active
 */
export function createVehicle(payload) {
  return http.post('/vehicles', payload).then((r) => r.data)
}

/** 차량 수정. */
export function updateVehicle(id, payload) {
  return http.put(`/vehicles/${id}`, payload).then((r) => r.data)
}

/** 차량 삭제. */
export function deleteVehicle(id) {
  return http.delete(`/vehicles/${id}`).then((r) => r.data)
}
